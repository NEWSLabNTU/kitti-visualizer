use crate::{
    read_pcd::{load_bin, InfoPoint},
    PcdFormat,
};
use anyhow::{Context, Result};
use kiss3d::{camera::Camera, text::Font, window::Window};
use kitti_format::{KittiCalib, KittiObject};
use nalgebra as na;
use std::{fs, io, path::Path, rc::Rc};

pub struct FrameData {
    pub objects: Vec<KittiObject>,
    pub points_in_range: Vec<InfoPoint>,
    pub points_out_range: Vec<InfoPoint>,
    pub num_points_map: Vec<usize>,
}

// pub fn put_text(image: &mut Mat, text: &str, org: cv_core::Point) {
//     imgproc::put_text(
//         image,
//         text,
//         org,
//         imgproc::FONT_HERSHEY_SIMPLEX,
//         1.0,
//         cv_core::Scalar::new(255.0, 191.0, 0.0, 0.0),
//         2,
//         imgproc::LINE_8,
//         false,
//     )
//     .unwrap();
// }

// pub fn draw_objects(
//     image: &mut Mat,
//     objects: &Vec<KittiObject>,
//     calib: &KittiCalib,
//     mode: &DrawingMode,
// ) {
//     for obj in objects {
//         match mode {
//             DrawingMode::Draw2D => {
//                 draw_2d_bbox(&obj, image);
//             }
//             DrawingMode::Draw3D => {
//                 let bbox = obj.bbox3d.clone();
//                 let vertices = bbox.vertices();
//                 let vertices_2d = project_3d_points(&vertices, calib);
//                 draw_bbox3d_edges(vertices_2d, image);
//             }
//         }
//     }
// }

// pub fn project_3d_points(
//     points3d: &Vec<na::Point3<f64>>,
//     calib: &KittiCalib,
// ) -> Vec<cv_core::Point2i> {
//     let mut points2d: Vec<cv_core::Point2i> = vec![];
//     let rect2velo = calib.get_transformation_from_rectified_camera_to_velodyne();
//     points3d.iter().for_each(|point3d| {
//         let rect_point = rect2velo.inverse() * point3d;
//         let rect_point_homo =
//             DMatrix::from_row_slice(4, 1, &[rect_point.x, rect_point.y, rect_point.z, 1.0]);
//         let camera_point = calib.p0.clone() * rect_point_homo;
//         let z = camera_point[(2, 0)];
//         let x = (camera_point[(0, 0)] / z) as i32;
//         let y = (camera_point[(1, 0)] / z) as i32;
//         points2d.push(cv_core::Point2i::new(x, y));
//     });
//     points2d
// }

// pub fn draw_bbox3d_edges(vertices: Vec<cv_core::Point2i>, image: &mut cv_core::Mat) {
//     let edge_relation = vec![
//         (0, 1),
//         (0, 2),
//         (1, 3),
//         (2, 3),
//         (4, 5),
//         (4, 6),
//         (5, 7),
//         (6, 7),
//         (0, 4),
//         (1, 5),
//         (2, 6),
//         (3, 7),
//         (1, 7),
//         (3, 5),
//     ];
//     edge_relation.into_iter().for_each(|(from_idx, to_idx)| {
//         imgproc::line(
//             image,
//             vertices[from_idx],
//             vertices[to_idx],
//             cv_core::Scalar::new(0.0, 255.0, 0.0, 0.0),
//             1,
//             8,
//             0,
//         )
//         .unwrap();
//     });
// }

pub fn get_indices_from_ann_dir(ann_dir: &Path) -> Vec<i32> {
    let mut entries = fs::read_dir(ann_dir)
        .unwrap()
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, io::Error>>()
        .unwrap();
    entries.sort();
    let indices: Vec<i32> = entries
        .iter()
        .filter_map(|path| {
            let path = path.as_path();
            let ext = path.extension();
            if ext.is_some() && ext.unwrap() == "txt" {
                let index_str = String::from(path.file_stem().unwrap().to_str().unwrap());
                let index = index_str.parse::<i32>().unwrap();
                Some(index)
            } else {
                None
            }
        })
        .collect();
    indices
}

// pub fn draw_2d_bbox(obj: &KittiObject, image: &mut cv_core::Mat) {
//     let bbox2d = &obj.bbox2d;
//     let bbox3d = &obj.bbox3d;
//     let rect = cv_core::Rect {
//         x: bbox2d.left as i32,
//         y: bbox2d.top as i32,
//         height: bbox2d.height as i32,
//         width: bbox2d.width as i32,
//     };

//     imgproc::rectangle(
//         image,
//         rect,
//         cv_core::Scalar::new(0.0, 255.0, 0.0, 0.0),
//         2, // thickness
//         imgproc::LINE_8,
//         0, // shift
//     )
//     .unwrap();

//     imgproc::put_text(
//         image,
//         &format!(
//             "{:.2} m",
//             na::Vector3::from([bbox3d.center().x, bbox3d.center().y, bbox3d.center().z]).norm()
//         ),
//         cv_core::Point2i::new(
//             (bbox2d.left + bbox2d.width / 2.) as i32,
//             (bbox2d.top + bbox2d.height / 2.) as i32,
//         ),
//         imgproc::FONT_HERSHEY_SIMPLEX,
//         0.5,
//         cv_core::Scalar::new(255.0, 191.0, 0.0, 0.0),
//         1,
//         imgproc::LINE_8,
//         false,
//     )
//     .unwrap();
// }

pub fn draw_objects_in_pcd(
    objects: &[KittiObject],
    num_points_map: &[usize],
    window: &mut Window,
    camera: &dyn Camera,
) {
    let edge_relation = vec![
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (1, 7),
        (3, 5),
    ];
    // let rect2velo = {
    //     let rect_to_cam = na::UnitQuaternion::from_matrix(&calib.r0_rect).inverse();
    //     let velo_to_cam_rot = na::UnitQuaternion::from_matrix(&na::Matrix3::from_row_slice(&[
    //         calib.velo_to_cam.m11,
    //         calib.velo_to_cam.m12,
    //         calib.velo_to_cam.m13,
    //         calib.velo_to_cam.m21,
    //         calib.velo_to_cam.m22,
    //         calib.velo_to_cam.m23,
    //         calib.velo_to_cam.m31,
    //         calib.velo_to_cam.m32,
    //         calib.velo_to_cam.m33,
    //     ]));
    //     let velo_to_cam_trans = na::Translation3::from([
    //         calib.velo_to_cam.m14,
    //         calib.velo_to_cam.m24,
    //         calib.velo_to_cam.m34,
    //     ]);
    //     let velo_to_cam = na::Isometry3::from_parts(velo_to_cam_trans, velo_to_cam_rot);
    //     velo_to_cam.inverse() * rect_to_cam
    // };
    for (_idx, obj) in objects.iter().enumerate() {
        let vertices = obj.bbox3d.vertices();
        edge_relation.iter().for_each(|(from_idx, to_idx)| {
            let color = na::Point3::from([0., 1., 0.]);
            window.draw_line(
                &vertices[*from_idx].cast::<f32>(),
                &vertices[*to_idx].cast::<f32>(),
                &color,
            );
        });
        // let num_points = num_points_map[idx];
        let text = format!("{:?}, {:.2}", obj.class.clone(), obj.bbox3d.extents.x);
        let color = if obj.object_key.is_some() {
            na::Point3::from([1., 0., 0.])
        } else {
            na::Point3::from([0., 0., 0.])
        };
        draw_text_3d(
            window,
            camera,
            &text,
            &na::Point3::cast(obj.bbox3d.pose.translation.vector.into()),
            50.0,
            &Font::default(),
            &color,
        );
    }
}

pub fn in_bbox(point: &na::Point3<f64>, objects: &[KittiObject]) -> bool {
    let mut result = false;
    for obj in objects {
        let bbox = &obj.bbox3d;
        let enlarge_offset = 0.1;
        let origin_point = bbox.pose.inverse() * point;
        if origin_point.x < bbox.extents.x / 2. + enlarge_offset
            && origin_point.x > -bbox.extents.x / 2. - enlarge_offset
            && origin_point.y < bbox.extents.y / 2. + enlarge_offset
            && origin_point.y > -bbox.extents.y / 2. - enlarge_offset
            && origin_point.z < bbox.extents.z / 2. + enlarge_offset
            && origin_point.z > -bbox.extents.z / 2. - enlarge_offset
        {
            result = true;
            break;
        }
    }

    result
}

fn draw_text_3d(
    window: &mut Window,
    camera: &dyn Camera,
    text: &str,
    pos: &na::Point3<f32>,
    scale: f32,
    font: &Rc<Font>,
    color: &na::Point3<f32>,
) {
    let window_size = na::Vector2::from([window.size()[0] as f32, window.size()[1] as f32]);
    let mut window_coord = camera.project(pos, &window_size);
    if window_coord.x.is_nan() || window_coord.y.is_nan() {
        return;
    }
    window_coord.y = window_size.y - window_coord.y;
    if window_coord.x >= window_size.x
        || window_coord.x < 0.0
        || window_coord.y >= window_size.y
        || window_coord.y < 0.0
    {
        return;
    }
    let coord: &na::Point2<f32> = &(window_coord * 2.0).into();
    window.draw_text(text, coord, scale, font, color);
}

pub fn get_objects_from_frame_id(
    index: i32,
    kitti_dir: &Path,
    supervisely_ann_dir: Option<&Path>,
    pcd_format: PcdFormat,
) -> Vec<KittiObject> {
    let objects = if supervisely_ann_dir.is_none() {
        let exclude_classes = vec!["DontCare".into()];
        let ann_dir = kitti_dir.join("label_2");
        let calib_dir = kitti_dir.join("calib");
        let ann_path = ann_dir.join(format!("{:0>6}.txt", index.to_string()));
        let calib_path = calib_dir.join(format!("{:0>6}.txt", index.to_string()));

        let calib = KittiCalib::from_file(calib_path);

        if pcd_format == PcdFormat::Philly {
            kitti_format::read_ann_file_philly(ann_path, &calib, &exclude_classes)
        } else {
            kitti_format::read_ann_file(ann_path, &calib, &exclude_classes)
        }
    } else {
        let ann_path = supervisely_ann_dir
            .as_ref()
            .unwrap()
            .join(format!("{:0>6}.pcd.json", index.to_string()));
        let objects: Vec<KittiObject> = kitti_format::read_from_supervisely(&ann_path);
        objects
    };
    objects
}

pub fn get_new_frame_data(
    index: i32,
    kitti_dir: &Path,
    supervisely_ann_dir: Option<&Path>,
    pcd_format: PcdFormat,
) -> Result<FrameData> {
    let pcd_dir = kitti_dir.join("velodyne");
    let objects = get_objects_from_frame_id(index, kitti_dir, supervisely_ann_dir, pcd_format);
    // let objects = index_to_objects.get(&index.unwrap()).unwrap();
    // Get the pcd file
    let pcd_path = pcd_dir.join(format!("{:0>6}.bin", index.to_string()));
    let info_points =
        load_bin(&pcd_path).with_context(|| format!("unable to read {}", pcd_path.display()))?;
    let points_in_range: Vec<_> = info_points
        .iter()
        .filter(|p| {
            in_bbox(
                &na::Point3::from([p.point.x, p.point.y, p.point.z]).cast(),
                &objects,
            )
        })
        .map(|p| (*p).clone())
        .collect();
    let points_out_range: Vec<_> = info_points
        .iter()
        .filter(|p| {
            !in_bbox(
                &na::Point3::from([p.point.x, p.point.y, p.point.z]).cast(),
                &objects,
            )
        })
        .map(|p| (*p).clone())
        .collect();
    let num_points_map: Vec<usize> = objects
        .iter()
        .map(|obj| {
            let num_points = info_points
                .iter()
                .filter(|p| {
                    in_bbox(
                        &na::Point3::from([p.point.x, p.point.y, p.point.z]).cast(),
                        &[obj.clone()],
                    )
                })
                .count();
            num_points
        })
        .collect();

    Ok(FrameData {
        objects,
        points_in_range,
        points_out_range,
        num_points_map,
    })
}
