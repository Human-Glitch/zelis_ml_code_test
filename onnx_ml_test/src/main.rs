use std::env;
use processor_utils::{preprocess_image, build_model, guess};

fn main() {
    let args: Vec<String> = env::args().collect();
    let img_path = &args[1];

    let ready_img = preprocess_image(img_path);
    let model = build_model();
    let guess = guess(&model, ready_img);

    print!("Guess: {}", guess);
}

mod processor_utils {
    use image::{imageops::{resize, FilterType}, open, DynamicImage, GrayImage, ImageReader};
    use ndarray::Array4;
    use ort::{inputs, GraphOptimizationLevel, Session};

    const MODEL_PATH: &str = "src/models/mnist_model.onnx";

    pub fn preprocess_image(img_path: &str) -> GrayImage {
        let img: DynamicImage = ImageReader::open(&img_path)
            .expect("File path did not contain an image.")
            .decode()
            .expect("Decode failed");

        let resized_img= resize(&img, 28, 28, FilterType::Nearest);
        let gray_img = DynamicImage::ImageRgba8(resized_img).to_luma8();
    
        return gray_img;
    }

    pub fn build_model() -> Session {
        return Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
            .with_intra_threads(4).unwrap()
            .commit_from_file(MODEL_PATH).unwrap();
    }

    pub fn guess(model: &Session, gray_img: GrayImage) -> f32 {
        let raw_pixels: Vec<u8> = gray_img
            .pixels()
            .map(|p| p[0])
            .collect();

        // Convert image into normalized ndarray
        let input = Array4::from_shape_vec(
            (1, 1, gray_img.height() as usize, gray_img.width() as usize),
            raw_pixels
                .into_iter()
                .map(|p| p as f32 / 255.).collect())
            .unwrap();

        let outputs = model
            .run(inputs![input.view()]
                .expect("Bad input!"))
                .expect("Could not run model.");

        let probabilities = &outputs[0]
            .try_extract_tensor::<f32>()
                .expect("Could not extract tensor float value.");

        for (index, &probability) in probabilities.iter().enumerate() {
            println!("Class Probabilities {}: {:.4}", index, probability);
        }

        let guess = probabilities.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        return guess as f32;
    }
}

// Stub out some tests
#[cfg(test)]
mod test {
    use std::any::{Any, TypeId};
    use ort::Session;
    use super::*;

    #[test]
    fn preprocess_image_when_handwritten_digit_5_jpg_then_is_resized_gray_scale_vector(){
        let img_path = "test_data/handwritten_5.jpg";

        let ready_img = preprocess_image(img_path);

        let (width, height) = ready_img.dimensions();
        assert_eq!(28, width);
        assert_eq!(28, height);
        for &value in ready_img.iter() {
            assert!(value >= 0 && value <= 255, "Value {} is out of bounds for grayscale!", value);
        }
    } 

    #[test]
    fn preprocess_image_when_handwritten_digit_3_png_then_is_resized_gray_scale_vector(){
        let img_path = "test_data/handwritten_3.png";

        let ready_img = preprocess_image(img_path);

        let (width, height) = ready_img.dimensions();
        assert_eq!(28, width);
        assert_eq!(28, height);
        for &value in ready_img.iter() {
            assert!(value >= 0 && value <= 255, "Value {} is out of bounds for grayscale!", value);
        }
    }

    #[test]
    fn build_model_when_using_mnist_onnx_model_then_return_session(){
        let model = build_model();

        assert_eq!(TypeId::of::<Session>(), model.type_id());
    }

    #[test]
    fn guess_when_handwritten_digit_5_jpg_then_guess_5(){
        let img_path = "test_data/handwritten_5.jpg";
        let ready_img = preprocess_image(img_path);
        let session = build_model();

        let res = guess(&session, ready_img);

        assert_eq!(res, 5.);
    }

    #[test]
    fn guess_when_handwritten_digit_0_jpeg_then_guess_0(){
        let img_path = "test_data/handwritten_0.jpeg";
        let ready_img = preprocess_image(img_path);
        let session = build_model();

        let res = guess(&session, ready_img);

        assert_eq!(res, 0.);
    }

    #[test]
    fn guess_when_handwritten_digit_1_jpeg_then_guess_1(){
        let img_path = "test_data/handwritten_1.jpeg";
        let ready_img = preprocess_image(img_path);
        let session = build_model();

        let res = guess(&session, ready_img);

        assert_eq!(res, 1.);
    }

    #[test]
    fn guess_when_handwritten_digit_2_jpeg_then_guess_2(){
        let img_path = "test_data/handwritten_2.jpeg";
        let ready_img = preprocess_image(img_path);
        let session = build_model();

        let res = guess(&session, ready_img);

        assert_eq!(res, 2.);
    }

    #[test]
    fn guess_when_handwritten_digit_4_jpeg_then_guess_4(){
        let img_path = "test_data/handwritten_4.jpeg";
        let ready_img = preprocess_image(img_path);
        let session = build_model();

        let res = guess(&session, ready_img);

        assert_eq!(res, 4.);
    }

    #[test]
    fn guess_when_handwritten_digit_7_jpeg_then_guess_7(){
        let img_path = "test_data/handwritten_7.jpeg";
        let ready_img = preprocess_image(img_path);
        let session = build_model();

        let res = guess(&session, ready_img);

        assert_eq!(res, 7.);
    }

    #[test]
    fn guess_when_handwritten_digit_8_jpeg_then_guess_8(){
        let img_path = "test_data/handwritten_8.jpeg";
        let ready_img = preprocess_image(img_path);
        let session = build_model();

        let res = guess(&session, ready_img);

        assert_eq!(res, 8.);
    }

    // #[test]
    // fn guess_when_handwritten_digit_3_png_then_label_3(){
    //     let img_path = "test_data/handwritten_3.png";
    //     let ready_img = preprocess_image(img_path);
    //     let session = build_model();

    //     let res = guess(&session, ready_img);

    //     assert_eq!(res, 3.);
    // }
}