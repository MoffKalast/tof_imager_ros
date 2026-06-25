import sys
import warnings
import rclpy
import numpy as np
from collections import deque

from typing import Optional
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from vl53l5cx.vl53l5cx import VL53L5CX as ToFImager, VL53L5CXResultsData as ToFImagerResults

class ScanDensifier:
	def __init__(self, num_beams, angle_min, angle_max, max_interp_angle, history_size=3, gradient_threshold=0.20):
		self.num_beams = num_beams
		self.angle_min = angle_min
		self.angle_max = angle_max
		self.angles = np.linspace(angle_min, angle_max, num_beams)
		self.history = deque(maxlen=history_size)
		# Bridge gaps up to a fixed angular width, converted to beams from the actual
		# beam pitch, so the reach is independent of num_beams.
		self.max_gap = int(round(max_interp_angle / ((angle_max - angle_min) / (num_beams - 1))))
		self.grad_thresh = gradient_threshold

	def update(self, zone_angles, rows):
		frame = np.full((rows.shape[0], self.num_beams), np.nan)
		for j, ang in enumerate(zone_angles):
			frame[:, np.argmin(np.abs(self.angles - ang))] = rows[:, j]
		self.history.append(frame)

		# Robust median over every available return (chosen rows x history). A lone
		# sun glint or flicker in one row/frame gets outvoted instead of dropped.
		stacked = np.concatenate(list(self.history), axis=0)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', category=RuntimeWarning)
			combined = np.nanmedian(stacked, axis=0)
		return self._interpolate(combined)

	def _interpolate(self, ranges):
		valid = ~np.isnan(ranges)
		if valid.sum() < 2:
			return ranges

		result = ranges.copy()
		indices = np.where(valid)[0]
		for i in range(len(indices) - 1):
			left, right = indices[i], indices[i+1]
			gap = right - left - 1
			if gap == 0 or gap > self.max_gap:
				continue
			r_left, r_right = ranges[left], ranges[right]
			if abs(r_right - r_left) / max(r_left, r_right, 0.001) > self.grad_thresh:
				continue  # edge detected, don't bridge it
			for j in range(1, gap + 1):
				t = j / (gap + 1)
				result[left + j] = r_left * (1 - t) + r_right * t
		return result

class ToFImagerPublisher(Node):
	def __init__(self):
		super().__init__('tof_imager')

		self.declare_parameters(namespace='', parameters=[
			('frame_id', 'tof_link'),
			('resolution', 8),
			('mode', 1),
			('ranging_freq', 15),
			('timer_period', 0.1),
			('fov_deg', 60.0),
			('scan_range_min', 0.02),
			('scan_range_max', 1.0),
			('scan_rows', [0, 1]),
			('valid_status', [5]),
			('max_ambient', 0.0),
			('num_beams', 50),
			('max_interp_zones', 1.2),
			('color_range', 0.2),
		])

		self.frame_id = self.get_parameter('frame_id').value
		self.res = self.get_parameter('resolution').value
		self.mode = self.get_parameter('mode').value
		self.freq = self.get_parameter('ranging_freq').value
		self.fov = np.deg2rad(self.get_parameter('fov_deg').value)
		self.range_min = self.get_parameter('scan_range_min').value
		self.range_max = self.get_parameter('scan_range_max').value
		self.scan_rows = list(self.get_parameter('scan_rows').value)
		self.valid_status = np.array(self.get_parameter('valid_status').value)
		self.max_ambient = self.get_parameter('max_ambient').value
		self.color_range = self.get_parameter('color_range').value

		# Beams sit at zone-center angles, not the FoV edges: the outermost zone center
		# is half a zone in from the edge, so the span is fov*(1 - 1/res).
		self.zone_angles = (np.arange(self.res) - (self.res-1)/2.0) * (self.fov/self.res)
		half_span = (self.fov/2.0) * (1.0 - 1.0/self.res)
		max_interp_angle = self.get_parameter('max_interp_zones').value * (self.fov/self.res)
		self.densifier = ScanDensifier(self.get_parameter('num_beams').value, -half_span, half_span, max_interp_angle)

		self.pcl_pub = self.create_publisher(PointCloud2, 'pointcloud', 10)
		self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)

		self.sensor: Optional[ToFImager] = None
		self.timer: Optional[Timer] = None

		self.init_sensor()
		self.timer = self.create_timer(self.get_parameter('timer_period').value, self.publish_data)

		self.get_logger().info('Node started')

	def init_sensor(self):
		self.sensor = ToFImager()
		self.sensor.init()

		if self.mode not in (1, 3):
			raise RuntimeError("Invalid mode")

		if self.res not in (4, 8):
			raise RuntimeError("Invalid resolution")

		self.sensor.set_resolution(self.res * self.res)

		freq = min(self.freq, 15) if self.res == 8 else min(self.freq, 60)
		self.sensor.set_ranging_frequency_hz(freq)
		self.sensor.set_ranging_mode(self.mode)
		self.sensor.start_ranging()

		if not self.sensor.is_alive():
			raise RuntimeError("Sensor not alive")

	def read_sensor(self):
		try:
			if not self.sensor.check_data_ready():
				return None
		except Exception:
			try:
				self.sensor.stop_ranging()
				self.sensor.start_ranging()
			except Exception:
				return None
			return None

		try:
			data = self.sensor.get_ranging_data()
		except IndexError:
			data = ToFImagerResults(nb_target_per_zone=1)
		except Exception:
			return None

		n = self.res * self.res
		distance_mm = np.array(data.distance_mm[:n]).reshape(self.res, self.res)
		depth_m = np.where(distance_mm < 0, 0, distance_mm).astype(np.float32) / 1000.0

		# distance is radial (a flat wall reads farther at the corners), so convert to
		# cartesian here: forward = d*cos(az)*cos(el) makes a flat wall land flat. az is
		# taken from the row index and el from the column, matching the original axes.
		per_px = self.fov / self.res
		buf = np.empty((self.res, self.res, 4), dtype=np.float32)
		for w in range(self.res):
			for h in range(self.res):
				d = depth_m[w, h]
				az = (w - (self.res-1)/2.0) * per_px
				el = (h - (self.res-1)/2.0) * per_px
				buf[w, h, :3] = [d*np.sin(az)*np.cos(el), -d*np.sin(el), d*np.cos(az)*np.cos(el)]

		# Packed rgb coloured by the z column: blue at 0, green at +range, red at -range.
		# Note z is forward depth here (always >= 0), so red never shows; set cval = y for
		# height colouring instead.
		cval = buf[:, :, 0]
		t_pos = np.clip(-cval / self.color_range, 0, 1)
		t_neg = np.clip(cval / self.color_range, 0, 1)
		r = (255 * t_neg).astype(np.uint32)
		g = (255 * t_pos).astype(np.uint32)
		b = (255 * np.clip(1 - t_pos - t_neg, 0, 1)).astype(np.uint32)
		buf[:, :, 3] = ((r << 16) | (g << 8) | b).view(np.float32)

		# Scan path: separate quality-masked depth so cloud output is unaffected.
		scan_m = depth_m.copy()
		scan_m[distance_mm < 0] = np.nan
		if self.valid_status.size and hasattr(data, 'target_status'):
			status = np.array(data.target_status[:n]).reshape(self.res, self.res)
			scan_m[~np.isin(status, self.valid_status)] = np.nan
		if self.max_ambient > 0 and hasattr(data, 'ambient_per_spad'):
			ambient = np.array(data.ambient_per_spad[:n]).reshape(self.res, self.res)
			scan_m[ambient > self.max_ambient] = np.nan

		return buf, depth_m, scan_m

	def publish_data(self):
		sensor_data = self.read_sensor()
		if sensor_data is None:
			return

		buf, depth_m, scan_m = sensor_data
		now = self.get_clock().now().to_msg()

		pc_msg = PointCloud2(
			header=Header(stamp=now, frame_id=self.frame_id),
			height=self.res,
			width=self.res,
			fields=[
				PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
				PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
				PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
				PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)
			],
			is_bigendian=False,
			is_dense=False,
			point_step=16,
			row_step=16 * self.res,
			data=buf.astype(np.float32).tobytes()
		)
		self.pcl_pub.publish(pc_msg)

		zone_angles = self.zone_angles[::-1]  # reversed to match sensor column order
		rows = scan_m[self.scan_rows, :].copy()
		rows[rows > self.range_max] = np.nan
		dense_ranges = self.densifier.update(zone_angles, rows)

		in_range = (dense_ranges >= self.range_min) & (dense_ranges <= self.range_max)
		dense_ranges = np.where(in_range, dense_ranges, np.inf)

		scan_msg = LaserScan(
			header=Header(stamp=now, frame_id='laser_link'),
			angle_min=self.densifier.angle_min,
			angle_max=self.densifier.angle_max,
			angle_increment=(self.densifier.angle_max - self.densifier.angle_min) / (self.densifier.num_beams-1),
			time_increment=0.0,
			scan_time=self.get_parameter('timer_period').value,
			range_min=self.range_min,
			range_max=self.range_max,
			ranges=dense_ranges.astype(np.float32).tolist(),
			intensities=[]
		)
		self.scan_pub.publish(scan_msg)


def main(args=None):
	rclpy.init(args=args)
	node = ToFImagerPublisher()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	except ExternalShutdownException:
		sys.exit(1)
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()