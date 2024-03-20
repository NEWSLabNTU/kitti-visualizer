use crate::{
    utils::{get_indices_from_ann_dir, get_new_frame_data, FrameData},
    PcdFormat,
};
use anyhow::{bail, Result};
use kiss3d::{
    camera::{ArcBall, Camera},
    event::{Action, Key, WindowEvent},
    planar_camera::PlanarCamera,
    post_processing::PostProcessingEffect,
    renderer::Renderer,
    text::Font,
    window::{State, Window},
};
use kiss3d_utils::WindowPlotExt;
use kitti_format::KittiObject;
use nalgebra as na;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use scarlet::{
    color::RGBColor,
    colormap::{ColorMap, ListedColorMap},
};
use std::{
    collections::HashMap,
    path::PathBuf,
    rc::Rc,
    sync::Once,
    time::{Duration, Instant},
};
use uluru::LRUCache;

const RANGE_META: [f32; 4] = [-30.0, 40.0, 40.4, 40.0];
static RANGE_VERTEX: Lazy<[na::Point3<f32>; 4]> = Lazy::new(|| {
    [
        na::Point3::from([RANGE_META[0], RANGE_META[1], 1.]),
        na::Point3::from([RANGE_META[0], RANGE_META[3], 1.]),
        na::Point3::from([RANGE_META[2], RANGE_META[3], 1.]),
        na::Point3::from([RANGE_META[2], RANGE_META[1], 1.]),
    ]
});
const FRAME_PERIOD: Duration = Duration::from_millis(100);

pub struct Gui {
    camera: ArcBall,
    options: GuiOptions,
    cache: GuiCache,
    data: GuiData,
}

type FrameIndex = usize;
type AnnotationIndex = usize;

struct GuiOptions {
    mark_points_in_boxes: bool,
    show_bbox: bool,
    play: bool,
    record: bool,
}

struct GuiCache {
    frame_idx: FrameIndex,
    frame_cache: HashMap<FrameIndex, FramePlot>,
    lru: LRUCache<FrameIndex, 32>,
    next_tick: Option<Instant>,
}

struct GuiData {
    indices: Vec<AnnotationIndex>,
    color_map: ListedColorMap,
    kitti_dir: PathBuf,
    supervisely_ann_dir: Option<PathBuf>,
    screencast_dir: Option<PathBuf>,
    pcd_format: PcdFormat,
}

struct FramePlot {
    points: Vec<PointPlot>,
    bboxes: Vec<BoxPlot>,
}

struct PointPlot {
    pos: na::Point3<f32>,
    color: na::Point3<f32>,
}

struct BoxPlot {
    box_edges: Vec<[na::Point3<f32>; 2]>,
    box_color: na::Point3<f32>,
    text: String,
    text_color: na::Point3<f32>,
    text_pos: na::Point3<f32>,
}

impl Gui {
    pub fn new(
        kitti_dir: PathBuf,
        supervisely_ann_dir: Option<PathBuf>,
        screencast_dir: Option<PathBuf>,
        pcd_format: PcdFormat,
        play_on_start: bool,
        record_on_start: bool,
    ) -> Result<Self> {
        let ann_dir = kitti_dir.join("label_2");
        let indices = get_indices_from_ann_dir(&ann_dir);

        if indices.is_empty() {
            bail!(
                "Unable to load annotation data from {}. Is it empty?",
                ann_dir.display()
            );
        }

        let record = match (record_on_start, screencast_dir.is_some()) {
            (true, true) => true,
            (true, false) => {
                static WARN: Once = Once::new();
                WARN.call_once(|| {
                    eprintln!("Warning: Request --record-on-start but --screencast-dir is not set");
                });
                false
            }
            (false, _) => false,
        };

        // let frame_data = get_new_frame_data(
        //     ann_idx as i32,
        //     &kitti_dir,
        //     supervisely_ann_dir.as_deref(),
        //     pcd_format,
        // );

        // let frame_data = match frame_data {
        //     Ok(frame) => {},
        //     Err(err) => {
        //         eprintln!("fail to load frame {ann_idx}: {err}");
        //         None
        //     }
        // };

        let camera = {
            let eye = na::Point3::from_slice(&[-20.0f32, -20.0, 20.0]);
            let at = na::Point3::from_slice(&[0.0f32, 0.0, 0.0]);
            let mut camera = ArcBall::new(eye, at);
            camera.set_up_axis(na::Vector3::from_column_slice(&[0., 0., 1.]));
            camera
        };

        let frame_idx = 0;

        let lru = LRUCache::default();

        Ok(Self {
            cache: GuiCache {
                frame_idx,
                frame_cache: HashMap::new(),
                lru,
                next_tick: None,
            },
            options: GuiOptions {
                mark_points_in_boxes: false,
                show_bbox: true,
                play: play_on_start,
                record,
            },
            data: GuiData {
                indices,
                color_map: ListedColorMap::plasma(),
                kitti_dir,
                supervisely_ann_dir,
                screencast_dir,
                pcd_format,
            },
            camera,
        })
    }

    fn render(&self, window: &mut Window) {
        let Self {
            cache: GuiCache { frame_idx, .. },
            data: GuiData { ref indices, .. },
            ..
        } = *self;
        let ann_idx = indices[frame_idx];

        // info_points.iter().for_each(|point| {
        //     let color = na::Point3::from([0.0, 0.0, 0.0]);
        //     window.draw_point(&point.point, &color)
        // });
        window.draw_text(
            &format!("frameID: {:?}", ann_idx),
            &na::Point2::from([0., 0.]),
            50.0,
            &Font::default(),
            &na::Point3::from([0., 0., 0.]),
        );

        for i in 0..4 {
            window.draw_line(
                &RANGE_VERTEX[i],
                &RANGE_VERTEX[(i + 1) % 4],
                &[0., 0., 0.].into(),
            );
        }

        window.draw_axes(na::Point3::origin(), 5.0);
        self.render_frame(window);
    }

    fn render_frame(&self, window: &mut Window) {
        let Self {
            cache:
                GuiCache {
                    frame_idx,
                    ref frame_cache,
                    ..
                },
            ..
        } = *self;

        let Some(frame) = frame_cache.get(&frame_idx) else {
            return;
        };
        self.draw_frame(frame, window);
    }

    fn draw_frame(&self, frame: &FramePlot, window: &mut Window) {
        let FramePlot { points, bboxes } = frame;

        for PointPlot { pos, color } in points {
            window.draw_point(pos, color)
        }

        if self.options.show_bbox {
            for bbox in bboxes {
                self.draw_bbox(bbox, window);
            }
        }
    }

    fn draw_bbox(&self, bbox: &BoxPlot, window: &mut Window) {
        let BoxPlot {
            box_edges,
            text,
            text_color,
            box_color,
            text_pos,
        } = bbox;

        for [p, q] in box_edges {
            window.draw_line(p, q, box_color);
        }

        self.draw_text_3d(window, text, text_pos, 50.0, &Font::default(), text_color);
    }

    fn draw_text_3d(
        &self,
        window: &mut Window,
        text: &str,
        pos: &na::Point3<f32>,
        scale: f32,
        font: &Rc<Font>,
        color: &na::Point3<f32>,
    ) {
        let window_size = na::Vector2::from([window.size()[0] as f32, window.size()[1] as f32]);
        let mut window_coord = self.camera.project(pos, &window_size);
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

    fn process_events(&mut self, window: &mut Window) {
        let Self {
            options:
                GuiOptions {
                    mark_points_in_boxes,
                    show_bbox,
                    play,
                    record,
                    ..
                },
            cache:
                GuiCache {
                    frame_idx,
                    next_tick,
                    ..
                },
            data,
            ..
        } = self;
        let GuiData {
            ref indices,
            ref screencast_dir,
            ..
        } = *data;

        let orig_frame_idx = *frame_idx;
        let mut new_frame_idx = *frame_idx;

        window.events().iter().for_each(|event| {
            use Action as A;
            use Key as K;
            use WindowEvent as E;

            match event.value {
                E::Key(K::R, A::Press, _) => {
                    if let (false, None) = (*record, screencast_dir) {
                        static WARN: Once = Once::new();
                        WARN.call_once(|| {
                            eprintln!("Warning: Request to record screencast but --screencast-dir is not set");
                        });
                    }

                    *record = !*record;
                }
                E::Key(K::I, A::Press, _) => {
                    *mark_points_in_boxes = !*mark_points_in_boxes;
                }
                E::Key(K::Space, A::Press, _) => {
                    *play = !*play;
                }
                E::Key(K::B, A::Press, _) => {
                    *show_bbox = !*show_bbox;
                }
                E::Key(K::Left, A::Press, _) => {
                    new_frame_idx = (new_frame_idx - 1).rem_euclid(indices.len());
                }
                E::Key(K::Right, A::Press, _) => {
                    new_frame_idx = (new_frame_idx + 1).rem_euclid(indices.len());
                }
                E::Key(K::Escape, A::Press, _) => {
                    window.close();
                }
                _ => {}
            }
        });

        if orig_frame_idx != new_frame_idx {
            *play = false;
        }

        if *play {
            let now = Instant::now();

            match next_tick {
                Some(next_tick) => {
                    if now >= *next_tick {
                        while now >= *next_tick {
                            *next_tick += FRAME_PERIOD;
                        }

                        if new_frame_idx + 1 < indices.len() {
                            new_frame_idx += 1;
                        } else {
                            *play = false;
                        }
                    }
                }
                None => {
                    *next_tick = Some(now + FRAME_PERIOD);
                }
            }
        }

        *frame_idx = new_frame_idx;
    }

    fn update(&mut self) {
        let Self {
            cache: GuiCache { frame_idx, .. },
            ..
        } = *self;

        use std::collections::hash_map::Entry;
        let Self {
            cache:
                GuiCache {
                    ref mut frame_cache,
                    ref mut lru,
                    ..
                },
            data:
                GuiData {
                    ref indices,
                    ref kitti_dir,
                    ref supervisely_ann_dir,
                    ref color_map,
                    pcd_format,
                    ..
                },
            options:
                GuiOptions {
                    mark_points_in_boxes: draw_in_intensity,
                    ..
                },
            ..
        } = *self;
        let ann_idx = indices[frame_idx];

        if let Entry::Vacant(entry) = frame_cache.entry(ann_idx) {
            let result = get_new_frame_data(
                ann_idx as i32,
                kitti_dir,
                supervisely_ann_dir.as_deref(),
                pcd_format,
            );

            let frame_data = match result {
                Ok(frame_data) => frame_data,
                Err(err) => {
                    eprintln!("fail to load frame {}: {err}", ann_idx);
                    return;
                }
            };

            let frame_plot = convert_frame(&frame_data, draw_in_intensity, color_map);

            entry.insert(frame_plot);

            if let Some(frame_idx_to_rm) = lru.insert(frame_idx) {
                self.cache.frame_cache.remove(&frame_idx_to_rm);
            }
        }
    }
}

impl State for Gui {
    fn step(&mut self, window: &mut Window) {
        self.process_events(window);
        self.update();
        self.render(window);

        match (self.options.record, &self.data.screencast_dir) {
            (true, Some(screencast_dir)) => {
                let screenshot = window.snap_image();
                let image_path = screencast_dir.join(format!("{:06}.jpg", self.cache.frame_idx));
                if let Err(err) = screenshot.save(&image_path) {
                    eprintln!("unable to save {}: {err}", image_path.display());
                }
            }
            _ => {}
        }
    }

    fn cameras_and_effect_and_renderer(
        &mut self,
    ) -> (
        Option<&mut dyn Camera>,
        Option<&mut dyn PlanarCamera>,
        Option<&mut dyn Renderer>,
        Option<&mut dyn PostProcessingEffect>,
    ) {
        (Some(&mut self.camera), None, None, None)
    }
}

fn convert_frame(
    frame_data: &FrameData,
    draw_in_intensity: bool,
    color_map: &ListedColorMap,
) -> FramePlot {
    // let Some(frame_data) = frame_cache.get(&frame_idx) else {
    //     return;
    // };

    let points_in_range = frame_data.points_in_range.par_iter().map(|point| {
        let color = if draw_in_intensity {
            let color: RGBColor = color_map.transform_single((point.intensity / 255. * 10.) as f64);
            na::Point3::from([color.r, color.g, color.b]).cast()
        } else {
            na::Point3::from([0.0, 0.0, 1.0])
        };
        (&point.point, color)
    });

    let points_out_range = frame_data.points_out_range.par_iter().map(|point| {
        let color = if draw_in_intensity {
            let color: RGBColor = color_map.transform_single((point.intensity / 255. * 10.) as f64);
            na::Point3::from([color.r, color.g, color.b]).cast()
        } else {
            na::Point3::from([0.0, 0.0, 1.0])
        };
        (&point.point, color)
    });

    let points: Vec<_> = points_in_range
        .chain(points_out_range)
        .map(|(&pos, color)| PointPlot { pos, color })
        .collect();

    let bboxes = convert_objects_in_pcd(&frame_data.objects, &frame_data.num_points_map);

    FramePlot { points, bboxes }
}

fn convert_objects_in_pcd(objects: &[KittiObject], _num_points_map: &[usize]) -> Vec<BoxPlot> {
    const EDGE_RELATION: &[(usize, usize)] = &[
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

    let box_plots: Vec<_> = objects
        .par_iter()
        .map(|obj| {
            let vertices = obj.bbox3d.vertices();

            let box_color: na::Point3<f32> = [0., 1., 0.].into();
            let box_edges: Vec<_> = EDGE_RELATION
                .par_iter()
                .copied()
                .map(|(from_idx, to_idx)| [vertices[from_idx].cast(), vertices[to_idx].cast()])
                .collect();

            // let num_points = num_points_map[idx];
            let text = format!("{:?}, {:.2}", obj.class.clone(), obj.bbox3d.extents.x);
            let text_color: na::Point3<f32> = if obj.object_key.is_some() {
                [1., 0., 0.]
            } else {
                [0., 0., 0.]
            }
            .into();
            let text_pos = na::Point3::from(obj.bbox3d.pose.translation.vector).cast();

            BoxPlot {
                box_edges,
                text,
                text_color,
                box_color,
                text_pos,
            }
        })
        .collect();

    box_plots
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
