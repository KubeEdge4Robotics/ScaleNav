import pyrealsense2 as rs
import numpy as np
import cv2 as cv

class RealSenseCameraDebug:
    def __init__(self, width=640, height=480, fps=30, name='D435'):
        self.name = name

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps) # depth frames configuration
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) # color frames configuration

        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name) == f'Intel RealSense {name}':
                config.enable_device(d.get_info(rs.camera_info.serial_number))
                break

        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.init_intrinsics()

    def get_intrinsics(self, frame):
        return frame.profile.as_video_stream_profile().intrinsics

    def init_intrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        self.color_intrinsics = self.get_intrinsics(color_frame)
        self.depth_intrinsics = self.get_intrinsics(depth_frame)

        self.camera_parameters = {
            'fx': self.color_intrinsics.fx,
            'fy': self.color_intrinsics.fy,
            'ppx': self.color_intrinsics.ppx,
            'ppy': self.color_intrinsics.ppy,
            'height': self.color_intrinsics.height,
            'width': self.color_intrinsics.width,
            'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
        }

    def get_image(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data()) # by default 16-bit

        return color_img, depth_img, depth_frame

    def get_3d_in_camera_frame_from_2d(self, coord_2d, depth_frame):
        distance = depth_frame.get_distance(*coord_2d)
        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, coord_2d, distance)
        return camera_coordinate


class RealSenseCamera(RealSenseCameraDebug):
    #def __init__(self, width=640, height=480, fps=30, name='D435'):
    def __init__(self, width=1280, height=720, fps=30, name='D455'):
        super().__init__(width, height, fps, name)
        #cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
        #cv.setMouseCallback(self.name, self.click)

    def click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print('clicked picture coord: ', x, y)
            color_img, depth_img, depth_frame = self.get_image()
            xyz = self.get_3d_in_camera_frame_from_2d((x, y), depth_frame)
            print('clicked camera coord: ', xyz)

    def get_image(self, show=False):
        color_img, depth_img, depth_frame = super().get_image()
        #if show:
            #cv.imshow(self.name, color_img)
           # cv.waitKey(1)
            #print(color_img.shape)
        return color_img, depth_img, depth_frame

class RealSenseCameraDepth(RealSenseCamera):
    def get_image(self):
        color_img, depth_img, depth_frame = super().get_image()

        depth_img_8bit = cv.convertScaleAbs(depth_img, alpha=0.05) # convert to 8-bit
        depth_colormap = cv.applyColorMap(depth_img_8bit, cv.COLORMAP_JET)
        # depth_img_3d = np.dstack([depth_img_8bit]*3)
        imgs = np.hstack([color_img, depth_colormap])
        
        
        cv.imshow(self.name, imgs)
        return color_img, depth_img, depth_frame



if __name__ == '__main__':

    auto_show = True
    show_depth = False

    if not auto_show:
        cam = RealSenseCameraDebug()
    else:
        if not show_depth:
            cam = RealSenseCamera()
        else:
            cam = RealSenseCameraDepth()

    if not auto_show:
        while True:
            color_img, depth_img, depth_frame = cam.get_image()
            
            cv.imshow(cam.name, color_img)
            if cv.waitKey(1) & 0xFF==ord('q'):
                break

    else:
        while True:
            color_img, depth_img, depth_frame = cam.get_image(True)
            if cv.waitKey(1) & 0xFF==ord('q'):
                break

    cv.destroyAllWindows()
