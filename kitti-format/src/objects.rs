use crate::{
    bbox::{BBox2D, BBox3D},
    calib::KittiCalib,
};
use nalgebra as na;
use std::{f64::consts::FRAC_PI_2, fs::File, io, io::BufRead, path::PathBuf};
use supervisely_format as sv;
use sv::Vector3D;

#[derive(Debug, Clone)]
pub struct KittiObject {
    pub class: String,
    pub bbox3d: BBox3D,
    pub bbox2d: BBox2D,
    pub score: Option<f64>,
    pub object_key: Option<String>,
}

impl KittiObject {
    pub fn is_scooter(&self) -> bool {
        self.class == "Cyclist" || self.class == "Scooter"
    }
}

pub fn read_from_supervisely(ann_dir: &PathBuf) -> Vec<KittiObject> {
    let sv::PointCloudAnnotation {
        figures, objects, ..
    } = serde_json::from_str(&std::fs::read_to_string(ann_dir).unwrap()).unwrap();

    let kitti_objects: Vec<KittiObject> = figures
        .iter()
        .enumerate()
        .map(|(_, figure)| {
            let super_object = objects
                .iter()
                .find(|obj| obj.key == figure.object_key)
                .unwrap();
            let sv::PointCloudFigure {
                geometry:
                    sv::PointCloudGeometry {
                        position: Vector3D { x, y, z },
                        rotation:
                            Vector3D {
                                x: rx,
                                y: ry,
                                z: rz,
                            },
                        dimensions:
                            Vector3D {
                                x: lx,
                                y: ly,
                                z: lz,
                            },
                    },
                ..
            } = *figure;
            let bbox3d = BBox3D {
                extents: na::Vector3::new(ly, lz, lx),
                pose: na::Isometry3::from_parts(
                    na::Translation3::new(x, y, z),
                    na::UnitQuaternion::from_euler_angles(rx, ry, rz + FRAC_PI_2),
                ),
            };
            let bbox2d = BBox2D::from_tlbr([0., 0., 0., 0.]);
            let confidence_tag = super_object
                .tags
                .iter()
                .find(|tag| tag.name == "Confidence");
            let confidence_score = if confidence_tag.is_some() {
                match confidence_tag.unwrap().value.as_ref().unwrap() {
                    sv::TagValue::Text(score) => Some(score.parse::<f64>().unwrap()),
                    _ => Some(1.0),
                }
            } else {
                Some(1.0)
            };
            KittiObject {
                bbox3d,
                bbox2d,
                class: super_object.class_title.clone(),
                score: confidence_score,
                object_key: Some(super_object.key.clone()),
            }
        })
        .collect();

    kitti_objects
}

pub fn read_ann_file_philly(
    ann_path: PathBuf,
    _calib: &KittiCalib,
    exclude_classes: &[String],
) -> Vec<KittiObject> {
    let file =
        File::open(ann_path.clone()).unwrap_or_else(|_| panic!("{:?} not exists!", ann_path));
    let content_lines = io::BufReader::new(file).lines();
    let mut objects: Vec<KittiObject> = vec![];
    // let rect2velo = calib.get_transformation_from_rectified_camera_to_velodyne();

    for line in content_lines {
        let line = line.unwrap();

        let words: Vec<&str> = line.split(' ').collect();
        let class = words[0].to_string();
        if exclude_classes.contains(&class) {
            continue;
        }
        let bbox3d = {
            let dimensions: Vec<f64> = words[8..=10]
                .iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            let locations: Vec<f64> = words[11..=13]
                .iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            let [lx, ly, lz] = dimensions[..] else {
                unreachable!()
            };
            let rect_center = na::Point3::from([locations[0], locations[1], locations[2]]);
            // let velo_center = rect2velo * rect_center;
            let rotation: f64 = words[14].parse().unwrap();
            // let z_rot = -rotation - PI / 2.;

            BBox3D {
                pose: na::Isometry3::from_parts(
                    rect_center.into(),
                    na::UnitQuaternion::from_euler_angles(0., 0., rotation),
                ),
                extents: [lx, ly, lz].into(),
            }
        };
        // dbg![&bbox3d];
        let bbox2d = {
            let ltrb: Vec<f64> = words[4..=7]
                .iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            BBox2D::from_tlbr([ltrb[1], ltrb[0], ltrb[3], ltrb[2]])
        };
        let score = if words.len() >= 16 {
            Some(words[15].parse::<f64>().unwrap())
        } else {
            None
        };
        let object = KittiObject {
            class,
            bbox3d,
            bbox2d,
            score,
            object_key: None,
        };
        objects.push(object);
    }
    objects
}

pub fn read_ann_file(
    ann_path: PathBuf,
    calib: &KittiCalib,
    exclude_classes: &[String],
) -> Vec<KittiObject> {
    let file =
        File::open(ann_path.clone()).unwrap_or_else(|_| panic!("{:?} not exists!", ann_path));
    let content_lines = io::BufReader::new(file).lines();
    let mut objects: Vec<KittiObject> = vec![];
    let rect2velo = calib.get_transformation_from_rectified_camera_to_velodyne();
    for line in content_lines {
        let line = line.unwrap();
        let words: Vec<&str> = line.split(' ').collect();
        let class = words[0].to_string();
        if exclude_classes.contains(&class) {
            continue;
        }
        let bbox3d = {
            let dimensions: Vec<f64> = words[8..=10]
                .iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            let locations: Vec<f64> = words[11..=13]
                .iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            let rect_center = na::Point3::from([
                locations[0],
                locations[1] - dimensions[0] / 2.,
                locations[2],
            ]);
            let velo_center = rect2velo * rect_center;
            let rotation = words[14].parse::<f64>().unwrap();
            let z_rot = -rotation - FRAC_PI_2;

            let [lx, ly, lz] = dimensions[..] else {
                unreachable!()
            };

            BBox3D {
                pose: na::Isometry3::from_parts(
                    velo_center.into(),
                    na::UnitQuaternion::from_euler_angles(0., 0., z_rot),
                ),
                extents: [lz, ly, lx].into(),
            }
        };
        let bbox2d = {
            let ltrb: Vec<f64> = words[4..=7]
                .iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            BBox2D::from_tlbr([ltrb[1], ltrb[0], ltrb[3], ltrb[2]])
        };
        let score = if words.len() >= 16 {
            Some(words[15].parse::<f64>().unwrap())
        } else {
            None
        };
        let object = KittiObject {
            class,
            bbox3d,
            bbox2d,
            score,
            object_key: None,
        };
        objects.push(object);
    }
    objects
}
