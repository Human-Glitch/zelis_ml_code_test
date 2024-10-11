use std::env;
use processor_utils::{preprocess_image, build_model, guess};

mod processor_utils;

fn main() {
    let args: Vec<String> = env::args().collect();
    let img_path = &args[1];

    let model = build_model();
    let gray_img = preprocess_image(img_path);
    let guess = guess(&model, &gray_img);

    println!("Guess: {}", &guess);
}