use nalgebra as na;
use std::{fs::File, io, io::BufRead, path::PathBuf};
#[derive(Clone)]
pub struct KittiCalib {
    // Coordinate transformation from rectified camera (3D) to camera image (2D)
    pub p0: na::Matrix3x4<f64>,
    // Coordinate transformation from Lidar to rectified camera
    pub velo_to_cam: na::Matrix3x4<f64>,
    // Coordinate transformation from rectified camera to camera
    pub r0_rect: na::Matrix3<f64>,
}

impl KittiCalib {
    pub fn from_file(calib_path: PathBuf) -> Self {
        let file = File::open(calib_path.clone()).expect(&format!("{:?} not exists!", calib_path));
        let content_lines = io::BufReader::new(file).lines();
        let mut p0 = na::Matrix3x4::default();
        let mut velo_to_cam = na::Matrix3x4::default();
        let mut r0_rect = na::Matrix3::default();
        for line in content_lines {
            if let Ok(line) = line {
                let mut words: Vec<&str> = line.split(&[' ', ':'][..]).collect();
                words.retain(|s| !s.is_empty());
                match words.first() {
                    Some(&"P0") => {
                        let vals: Vec<f64> = words[1..]
                            .iter()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect();
                        let mat = na::Matrix3x4::from_row_slice(&vals);
                        p0 = mat;
                    }
                    Some(&"Tr_velo_to_cam") => {
                        let vals: Vec<f64> = words[1..]
                            .iter()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect();
                        let mat = na::Matrix3x4::from_row_slice(&vals);
                        velo_to_cam = mat;
                    }
                    Some(&"R0_rect") => {
                        let vals: Vec<f64> = words[1..]
                            .iter()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect();
                        let mat = na::Matrix3::from_row_slice(&vals);
                        r0_rect = mat;
                    }
                    _ => {}
                }
            }
        }
        KittiCalib {
            p0,
            velo_to_cam,
            r0_rect,
        }
    }
    pub fn get_transformation_from_rectified_camera_to_velodyne(&self) -> na::Isometry3<f64> {
        let rect2velo = {
            let rect_to_cam = na::UnitQuaternion::from_matrix(&self.r0_rect).inverse();
            let velo_to_cam_rot = na::UnitQuaternion::from_matrix(&na::Matrix3::from_row_slice(&[
                self.velo_to_cam.m11,
                self.velo_to_cam.m12,
                self.velo_to_cam.m13,
                self.velo_to_cam.m21,
                self.velo_to_cam.m22,
                self.velo_to_cam.m23,
                self.velo_to_cam.m31,
                self.velo_to_cam.m32,
                self.velo_to_cam.m33,
            ]));
            let velo_to_cam_trans = na::Translation3::from([
                self.velo_to_cam.m14,
                self.velo_to_cam.m24,
                self.velo_to_cam.m34,
            ]);
            let velo_to_cam = na::Isometry3::from_parts(velo_to_cam_trans, velo_to_cam_rot);
            velo_to_cam.inverse() * rect_to_cam
        };
        rect2velo
    }
}
