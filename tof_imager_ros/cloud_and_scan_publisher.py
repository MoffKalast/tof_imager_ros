import sys
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
    def __init__(self, num_beams=32, fov_deg=60, history_size=3, max_interp_gap=4, gradient_threshold=0.20):
        self.num_beams = num_beams
        self.fov = np.deg2rad(fov_deg)
        self.history = deque(maxlen=history_size)
        self.max_gap = max_interp_gap  # in "virtual beam" units
        self.grad_thresh = gradient_threshold  # relative (10%)
        
        # Virtual beam angles
        self.angles = np.linspace(-self.fov/2, self.fov/2, num_beams)
        
    def update(self, raw_angles, row1_ranges, row2_ranges):
        """
        raw_angles: beam angles (length 8)
        row1_ranges: top row ranges (length 8)
        row2_ranges: second row ranges (length 8)
        """
        # 1. Map both rows to virtual grid for this frame
        frame = np.full((2, self.num_beams), np.nan)
        for ang, rng1, rng2 in zip(raw_angles, row1_ranges, row2_ranges):
            idx = np.argmin(np.abs(self.angles - ang))
            frame[0, idx] = rng1
            frame[1, idx] = rng2
        
        self.history.append(frame)
        
        # 2. Median across time AND both rows (2 rows x up to 3 frames = up to 6 values)
        if len(self.history) < 2:
            return frame[0, :]  # Not enough history yet
        
        stacked = np.stack(self.history, axis=0)  # (H, 2, N)
        all_values = stacked.reshape(-1, self.num_beams)  # (H*2, N)
        temporal = np.nanmedian(all_values, axis=0)  # (N,)
        
        # 3. Spatial interpolation with edge awareness
        return self._interpolate(temporal)

    """ def update(self, raw_angles, raw_ranges):
        # 1. Temporal accumulation: map raw beams to virtual grid, store history
        frame = np.full(self.num_beams, np.nan)
        for ang, rng in zip(raw_angles, raw_ranges):
            idx = np.argmin(np.abs(self.angles - ang))
            frame[idx] = rng
        
        self.history.append(frame)
        
        # 2. Median filter across time (per beam)
        if len(self.history) < 2:
            return frame  # Not enough history yet
        
        temporal = np.nanmedian(np.stack(self.history), axis=0)
        
        # 3. Spatial interpolation with edge awareness
        return self._interpolate(temporal) """
    
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
            # Relative gradient check
            if abs(r_right - r_left) / max(r_left, r_right, 0.001) > self.grad_thresh:
                continue  # Edge detected, don't interpolate
            
            # Linear interpolation (or cubic if gap is large enough)
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
		])

		self.densifier = ScanDensifier()

		self.frame_id = self.get_parameter('frame_id').value
		self.res = self.get_parameter('resolution').value
		self.mode = self.get_parameter('mode').value
		self.freq = self.get_parameter('ranging_freq').value

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

		distance_mm = np.array(data.distance_mm[:(self.res*self.res)]).reshape(self.res, self.res)

		depth_m = np.where(distance_mm < 0, 0, distance_mm).astype(np.float32) / 1000.0
		depth_mm = np.where(distance_mm < 0, 0, distance_mm).astype(np.uint16)

		buf = np.empty((self.res, self.res, 3), dtype=np.float32)
		per_px = np.deg2rad(45) / self.res

		for w in range(self.res):
			for h in range(self.res):
				d = depth_m[w, h]
				x = d * np.cos(w*per_px - np.deg2rad(45)/2 - np.deg2rad(90))
				y = -(d * np.sin(h*per_px - np.deg2rad(45)/2))
				z = d
				buf[w, h] = [x, y, z]

		return buf, depth_m, depth_mm

	def publish_data(self):
		sensor_data = self.read_sensor()
		if sensor_data is None:
			return

		buf, depth_m, depth_mm = sensor_data
		now = self.get_clock().now().to_msg()

		pc_msg = PointCloud2(
			header=Header(stamp=now, frame_id=self.frame_id),
			height=self.res,
			width=self.res,
			fields=[
				PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
				PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
				PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
			],
			is_bigendian=False,
			is_dense=False,
			point_step=12,
			row_step=12 * self.res,
			data=buf.astype(np.float32).tobytes()
		)
		self.pcl_pub.publish(pc_msg)

		# Use raw depth_m from the top row as ranges — buf contains projected XYZ so
		# norm() on those would give distorted distances. The 27deg pitch offset from
		# boresight to the top row center is handled by laser_link's TF in the URDF.
		fov = np.deg2rad(60.0)
		raw_angles = np.linspace(-fov / 2, fov / 2, self.res)
		raw_angles_reversed = raw_angles[::-1]
		#dense_ranges = self.densifier.update(raw_angles_reversed, depth_m[0, :])
		top_row = depth_m[0, :]
		second_row = depth_m[1, :]
		dense_ranges = self.densifier.update(raw_angles_reversed, top_row, second_row)

		scan_msg = LaserScan(
			header=Header(stamp=now, frame_id='laser_link'),
			angle_min=-fov / 2,
			angle_max=fov / 2,
			angle_increment=fov / (self.densifier.num_beams-1),
			time_increment=0.0,
			scan_time=self.get_parameter('timer_period').value,
			range_min=0.02,
			range_max=4.0,
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