extern crate pretty_env_logger;
#[macro_use] extern crate log;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{EventLoop, ControlFlow},
    window::WindowBuilder,
};

fn main() {
    pretty_env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let _window = WindowBuilder::new()
        .with_title("example1")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();

    let window2 = WindowBuilder::new()
        .with_title("example2")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();
    let target_window_id = window2.id();


    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                window_id: target_window_id,
                event: WindowEvent::CloseRequested,
            } => {
                debug!("The close button was pressed; stopping");
                elwt.exit();
            }

            _ => {}
        }
    });

}
