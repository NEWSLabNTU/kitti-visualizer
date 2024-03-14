mod read_pcd;
mod utils;

use crate::utils::*;
use clap::{Parser, ValueEnum};
use kiss3d::{
    camera::ArcBall,
    event::{Action, Key, WindowEvent},
    window::Window,
};
use kiss3d_utils::*;
use nalgebra as na;
use scarlet::colormap::ColorMap;
use std::path::PathBuf;

#[derive(Parser)]
struct Opts {
    #[clap(short, long)]
    pub kitti_dir: PathBuf,
    #[clap(short, long)]
    pub supervisely_ann_dir: Option<PathBuf>,
    #[clap(short, long, default_value = "libpcl")]
    pub format: PcdFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum PcdFormat {
    Libpcl,
    Philly,
}

fn main() {
    let Opts {
        kitti_dir,
        supervisely_ann_dir,
        format,
    } = Opts::parse();

    visualize_in_pcd(kitti_dir, supervisely_ann_dir, format);
}

fn visualize_in_pcd(
    kitti_dir: PathBuf,
    supervisely_ann_dir: Option<PathBuf>,
    pcd_format: PcdFormat,
) {
    let ann_dir = kitti_dir.join("label_2");
    let indices = get_indices_from_ann_dir(&ann_dir);
    let mut idx = 0;
    let mut index: Option<i32> = Some(indices[idx]);
    let eye = na::Point3::from_slice(&[30.0f32, 0.0, 70.0]);
    let at = na::Point3::from_slice(&[30.0f32, 0.0, 0.0]);
    let mut camera = ArcBall::new(eye, at);
    camera.set_up_axis(na::Vector3::from_column_slice(&[0., 0., 1.]));
    let mut draw_in_intensity = false;
    let color_map = scarlet::colormap::ListedColorMap::plasma();
    println!("Reading objects...");

    println!("Start Rendering...");
    loop {
        if index.is_none() {
            eprint!("Enter the dataset index: ");
            index = Some(text_io::read!());
            idx = indices.iter().position(|&i| Some(i) == index).unwrap();
        }
        let (mut objects, mut points_in_range, mut points_out_range, mut num_points_map) =
            get_new_frame_data(
                index.unwrap(),
                &kitti_dir,
                supervisely_ann_dir.as_deref(),
                pcd_format,
            );
        let range_meta = {
            let x_min = -30 as f32;
            let x_max = 40.4 as f32;
            let y_min = -40 as f32;
            let y_max = 40 as f32;
            (x_min, y_min, x_max, y_max)
        };
        let range_vertex = vec![
            na::Point3::from([range_meta.0, range_meta.1, 1.]),
            na::Point3::from([range_meta.0, range_meta.3, 1.]),
            na::Point3::from([range_meta.2, range_meta.3, 1.]),
            na::Point3::from([range_meta.2, range_meta.1, 1.]),
        ];

        // Draw in window
        {
            let mut window = Window::new_with_size("debug", 1920, 1080);
            window.set_background_color(1., 1., 1.);
            window.set_line_width(2.);

            while window.render_with_camera(&mut camera) {
                // info_points.iter().for_each(|point| {
                //     let color = na::Point3::from([0.0, 0.0, 0.0]);
                //     window.draw_point(&point.point, &color)
                // });
                window.draw_text(
                    &format!("frameID: {:?}", index.unwrap()),
                    &na::Point2::from([0., 0.]),
                    50.0,
                    &kiss3d::text::Font::default(),
                    &na::Point3::from([0., 0., 0.]),
                );

                for i in 0..4 {
                    window.draw_line(
                        &range_vertex[i],
                        &range_vertex[(i + 1) % 4],
                        &na::Point3::from([0., 0., 0.]),
                    );
                }

                window.draw_axes(na::Point3::origin(), 5.0);
                points_in_range.iter().for_each(|point| {
                    let color = if draw_in_intensity {
                        let color: scarlet::color::RGBColor =
                            color_map.transform_single((point.intensity / 255. * 10.) as f64);
                        na::Point3::from([color.r, color.g, color.b]).cast()
                    } else {
                        na::Point3::from([0.0, 0.0, 1.0])
                    };
                    window.draw_point(&point.point, &color)
                });
                // if draw_filtered_points {
                points_out_range.iter().for_each(|point| {
                    let color = if draw_in_intensity {
                        let color: scarlet::color::RGBColor =
                            color_map.transform_single((point.intensity / 255. * 10.) as f64);
                        let color = na::Point3::from([color.r, color.g, color.b]).cast();
                        color
                    } else {
                        na::Point3::from([0.0, 0.0, 1.0])
                    };
                    window.draw_point(&point.point, &color)
                });
                // }
                draw_objects_in_pcd(&objects, &num_points_map, &mut window, &camera);
                window.events().iter().for_each(|event| {
                    use Action as A;
                    use Key as K;
                    use WindowEvent as E;

                    match event.value {
                        E::Key(K::I, A::Press, _) => {
                            draw_in_intensity = !draw_in_intensity;
                        }
                        E::Key(K::Left, A::Press, _) => {
                            idx = (idx - 1).rem_euclid(indices.len());
                            index = Some(indices[idx]);

                            let new_frame_data = get_new_frame_data(
                                index.unwrap(),
                                &kitti_dir,
                                supervisely_ann_dir.as_deref(),
                                pcd_format,
                            );

                            objects = new_frame_data.0;
                            points_in_range = new_frame_data.1;
                            points_out_range = new_frame_data.2;
                            num_points_map = new_frame_data.3;
                        }
                        E::Key(K::Right, A::Press, _) => {
                            idx = (idx + 1).rem_euclid(indices.len());
                            index = Some(indices[idx]);
                            let new_frame_data = get_new_frame_data(
                                index.unwrap(),
                                &kitti_dir,
                                supervisely_ann_dir.as_deref(),
                                pcd_format,
                            );

                            objects = new_frame_data.0;
                            points_in_range = new_frame_data.1;
                            points_out_range = new_frame_data.2;
                            num_points_map = new_frame_data.3;
                        }
                        E::Key(K::Escape, A::Press, _) => {
                            index = None;
                            window.close();
                        }

                        _ => {}
                    }
                });
            }
        }
    }
}

// fn visualize_in_image(kitti_dir: PathBuf) {
//     let exclude_classes = vec!["DontCare".into()];
//     let image_dir = kitti_dir.join("image_2");
//     let calib_dir = kitti_dir.join("calib");
//     let ann_dir = kitti_dir.join("label_2");
//     let indices = get_indices_from_ann_dir(&ann_dir);
//     let mut idx = 0;
//     let mut index: Option<i32> = Some(indices[idx]);
//     let mut mode = DrawingMode::Draw3D;

//     loop {
//         if index.is_none() {
//             eprint!("Enter the dataset index: ");
//             index = Some(read!());
//             idx = indices.iter().position(|&i| Some(i) == index).unwrap();
//         }
//         // Get the calib file
//         let calib_path = calib_dir.join(format!("{:0>6}.txt", index.unwrap().to_string()));
//         let calib = KittiCalib::from_file(calib_path);
//         // Get the annotation file
//         let ann_path = ann_dir.join(format!("{:0>6}.txt", index.unwrap().to_string()));
//         let objects = kitti_format::read_ann_file(ann_path, &calib, &exclude_classes);
//         let image_path = image_dir.join(format!("{:0>6}.png", index.unwrap().to_string()));
//         let mut image = imread(image_path.to_str().unwrap(), 1).unwrap();

//         // Draw bboxes
//         draw_objects(&mut image, &objects, &calib, &mode);
//         put_text(
//             &mut image,
//             &format!(
//                 "frame index: {}",
//                 format!("{:0>6}", index.unwrap().to_string())
//             ),
//             cv_core::Point::new(0, 30),
//         );
//         imshow("image", &image).unwrap();
//         loop {
//             let key = wait_key(0).unwrap();
//             if key == 27 {
//                 index = None;
//                 break;
//             }
//             if key == 81 {
//                 if idx == 0 {
//                     idx = indices.len() - 1;
//                 } else {
//                     idx = idx - 1;
//                 }

//                 index = Some(indices[idx]);
//                 break;
//             }
//             if key == 83 {
//                 idx = idx + 1;
//                 if idx >= indices.len() {
//                     idx = 0;
//                 }
//                 index = Some(indices[idx]);
//                 break;
//             }
//             if key == 115 {
//                 match mode {
//                     DrawingMode::Draw3D => {
//                         mode = DrawingMode::Draw2D;
//                     }
//                     DrawingMode::Draw2D => {
//                         mode = DrawingMode::Draw3D;
//                     }
//                 }
//                 break;
//             }
//             // if key == 114 {
//             //     objects
//             //         .iter_mut()
//             //         .for_each(|obj| (*obj).bbox3d.rot_y += 0.1);
//             //     break;
//             // }
//         }
//     }
// }
