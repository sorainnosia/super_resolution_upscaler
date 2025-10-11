// Add to Cargo.toml:
/*
[dependencies]
iced = { version = "0.12", features = ["image", "tokio"] }
ort = { version = "2.0.0-rc.4", features = ["load-dynamic"] }
ndarray = "0.16"
image = "0.25"
anyhow = "1.0"
reqwest = { version = "0.12", features = ["blocking"] }
tokio = { version = "1", features = ["full"] }
rfd = "0.14"
rayon = "1.10"
num_cpus = "1.16"
*/

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use iced::{
    executor, font, theme,
    widget::{button, column, container, pick_list, row, text, scrollable, Space, image as iced_image},
    Alignment, Application, Color, Command, Element, Font, Length, Settings, Size, Theme, Background,
};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array4;
use ort::{session::Session, value::Value};
use std::path::{Path, PathBuf};
use std::{fs, io};
use std::time::Duration;
use std::sync::Arc;
use anyhow::Result;
use iced::widget::scrollable::{Direction, Properties};
use std::process::{Command as ProcessCommand, Stdio};
use std::io::Write;
use rayon::prelude::*;

use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use chrono::Local;

// Font definitions
const HEADING_FONT: Font = Font {
    family: font::Family::Name("Noto Sans"),
    weight: font::Weight::Bold,
    stretch: font::Stretch::Normal,
    style: font::Style::Normal,
};

const BODY_FONT: Font = Font {
    family: font::Family::Name("Noto Sans"),
    weight: font::Weight::Normal,
    stretch: font::Stretch::Normal,
    style: font::Style::Normal,
};

// Theme colors
const PRIMARY_COLOR: Color = Color::from_rgb(0.2, 0.5, 0.9);
const BACKGROUND_COLOR: Color = Color::from_rgb(0.97, 0.97, 0.98);
const CARD_COLOR: Color = Color::WHITE;
const TEXT_COLOR: Color = Color::from_rgb(0.2, 0.2, 0.3);
const TEXT_SECONDARY: Color = Color::from_rgb(0.4, 0.4, 0.5);

pub fn main() -> iced::Result {
    let mut settings = Settings::default();
    settings.window.size = Size::new(1200.0, 800.0);
    settings.fonts = vec![
        include_bytes!("../assets/NotoSans-Regular.ttf").as_slice().into(),
        include_bytes!("../assets/NotoSans-Bold.ttf").as_slice().into(),
    ];
    settings.default_font = BODY_FONT;
    settings.default_text_size = 14.into();
    App::run(settings)
}

#[derive(Debug, Clone)]
enum Message {
    BrowseFile,
    BrowseFolder,
    FileSelected(Option<PathBuf>),
    FolderSelected(Option<PathBuf>),
    CategorySelected(ModelType),
    ModelSelected(ModelInfo),
    PreviewFileSelected(String),
    Process,
    ProcessComplete(Result<Vec<ProcessResult>, String>),
    PreviewLoaded(Result<(DynamicImage, PathBuf), String>),
    ZoomIn,
    ZoomOut,
    ResetZoom,
    BrowseVideo,
    VideoSelected(Option<PathBuf>),
    ProcessVideo,
    VideoProcessComplete(Result<String, String>),
}

struct App {
    input_path: Option<PathBuf>,
    input_type: InputType,
    available_models: Vec<ModelInfo>,
    selected_category: Option<ModelType>,
    selected_model: Option<ModelInfo>,
    image_files: Vec<PathBuf>,
    selected_preview_file: Option<String>,
    before_image: Option<Arc<DynamicImage>>,
    after_image: Option<Arc<DynamicImage>>,
    process_results: Vec<ProcessResult>,
    processing: bool,
    status_message: String,
    zoom_level: f32,
}

#[derive(Debug, Clone, PartialEq)]
enum InputType {
    None,
    File,
    Folder,
    Video
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ModelType {
    Upscaling,
    Denoising,
    Enhancement,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Upscaling => write!(f, "Upscaling"),
            ModelType::Denoising => write!(f, "Denoising"),
            ModelType::Enhancement => write!(f, "Enhancement"),
        }
    }
}

// Add this enum near ModelType
#[derive(Debug, Clone, PartialEq)]
enum TensorFormat {
    NCHW, // Standard: [batch, channels, height, width]
    NHWC, // Alternative: [batch, height, width, channels]
}

// Add these enums near ModelType
#[derive(Debug, Clone, PartialEq)]
enum NormalizationRange {
    ZeroOne,      // [0, 1]
    MinusOneOne,  // [-1, 1]
}

#[derive(Debug, Clone, PartialEq)]
struct ModelInfo {
    name: String,
    url: String,
    model_type: ModelType,
    scale: u32,
    window_size: u32,
    description: String,
    category: String,
	tensor_format: TensorFormat, // NEW FIELD
    input_norm: NormalizationRange,  // NEW: Input normalization
    output_norm: NormalizationRange,
	min_dimension: Option<u32>, // NEW: Minimum width/height requirement
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.model_type {
            ModelType::Upscaling => write!(f, "{} - {} ({}x)", self.category, self.description, self.scale),
            ModelType::Denoising | ModelType::Enhancement => write!(f, "{} - {}", self.category, self.description),
        }
    }
}

#[derive(Debug, Clone)]
struct ProcessResult {
    input_path: PathBuf,
    output_path: PathBuf,
    input_dims: (u32, u32),
    output_dims: (u32, u32),
    duration: f32,
}

impl Application for App {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let models = vec![
            // ===== UPSCALING MODELS =====
            ModelInfo {
                name: "swin2SR-realworld-sr-x4-64-bsrgan-psnr".to_string(),
                url: "https://huggingface.co/Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 8,
                description: "Real-world photos (4x)".to_string(),
                category: "Swin2SR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "swin2SR-classical-sr-x4-64".to_string(),
                url: "https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 8,
                description: "Clean images (4x)".to_string(),
                category: "Swin2SR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "swin2SR-lightweight-x2-64".to_string(),
                url: "https://huggingface.co/Xenova/swin2SR-lightweight-x2-64/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 2,
                window_size: 8,
                description: "Lightweight (2x)".to_string(),
                category: "Swin2SR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "swin2SR-compressed-sr-x4-48".to_string(),
                url: "https://huggingface.co/Xenova/swin2SR-compressed-sr-x4-48/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 8,
                description: "Compressed/JPEG (4x)".to_string(),
                category: "Swin2SR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "2x_APISR_RRDB_GAN_generator".to_string(),
                url: "https://huggingface.co/Xenova/2x_APISR_RRDB_GAN_generator-onnx/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 2,
                window_size: 1,
                description: "APISR GAN (2x) Anime".to_string(),
                category: "APISR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "4x_APISR_GRL_GAN_generator".to_string(),
                url: "https://huggingface.co/Xenova/4x_APISR_GRL_GAN_generator-onnx/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 1,
                description: "APISR GAN (4x) Anime".to_string(),
                category: "APISR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            
            // ===== RESTORATION & ENHANCEMENT MODELS (TensorStack) =====
            ModelInfo {
                name: "SwinIR-Noise".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/SwinIR-Noise/model.onnx".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 8,
                description: "Noise reduction".to_string(),
                category: "SwinIR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "SwinIR-BSRGAN-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/SwinIR-BSRGAN-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 8,
                description: "Real degradations (4x)".to_string(),
                category: "SwinIR".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "BSRGAN-2x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/BSRGAN-2x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 2,
                window_size: 1,
                description: "Blind SR (2x)".to_string(),
                category: "BSRGAN".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "RealESRGAN-2x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/RealESRGAN-2x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 2,
                window_size: 1,
                description: "Real-world SR (2x)".to_string(),
                category: "RealESRGAN".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "RealESRGAN-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/RealESRGAN-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "Real-world SR (4x)".to_string(),
                category: "RealESRGAN".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "RealESR-General-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/RealESR-General-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "General purpose (4x)".to_string(),
                category: "RealESRGAN".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "Swin2SR-Classical-2x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/Swin2SR-Classical-2x/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 2,
                window_size: 8,
                description: "Classical SR (2x)".to_string(),
                category: "Swin2SR-TS".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "Swin2SR-Classical-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/Swin2SR-Classical-4x/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 8,
                description: "Classical SR (4x)".to_string(),
                category: "Swin2SR-TS".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "UltraSharp-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/UltraSharp-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "Ultra sharp details (4x)".to_string(),
                category: "Custom".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "UltraMix-Smooth-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/UltraMix-Smooth-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "Ultra smooth details (4x)".to_string(),
                category: "Custom".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
			ModelInfo {
                name: "denoiser".to_string(),
                url: "denoiser".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "(Train)".to_string(),
                category: "Denoiser".to_string(),
				tensor_format: TensorFormat::NHWC,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "deblurring_nafnet_2025may".to_string(),
                url: "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 512,
                description: "Motion deblur (GoPro)".to_string(),
                category: "NAFNet - Motion deblur".to_string(),
				tensor_format: TensorFormat::NCHW,							
				input_norm: NormalizationRange::ZeroOne,  // Input: [-1, 1]
				output_norm: NormalizationRange::ZeroOne,     // Output: [0, 1]
				min_dimension: Some(512),
            },
			ModelInfo {
				name: "deblurgan_mobilenet".to_string(),
				url: "local".to_string(),
				model_type: ModelType::Denoising,
				scale: 1,
				window_size: 16,
				description: "Motion deblur (fast)".to_string(),
				category: "DeblurGAN-v2".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,  // Input: [-1, 1]
				output_norm: NormalizationRange::ZeroOne,     // Output: [0, 1] ← FIX
				min_dimension: None, // No minimum for most models
			},
            ModelInfo {
                name: "restormer_deraining".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer deraining".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_real".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (real)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_defocus_dual".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer defocus (dual)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_defocus_single".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer defocus (single)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_color_blind".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (color blind)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_color_sigma15".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (color sigma15)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_color_sigma25".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (color sigma25)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_color_sigma50".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (color sigma50)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_gray_blind".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (gray blind)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_gray_sigma15".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (gray sigma15)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_gray_sigma25".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (gray sigma25)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            },
            ModelInfo {
                name: "restormer_denoising_gray_sigma50".to_string(),
                url: "local".to_string(),
                model_type: ModelType::Denoising,
                scale: 1,
                window_size: 64,
                description: "Restormer denoising (gray sigma50)".to_string(),
                category: "NAFNet".to_string(),
				tensor_format: TensorFormat::NCHW,
				input_norm: NormalizationRange::ZeroOne,
				output_norm: NormalizationRange::ZeroOne,
				min_dimension: None, // No minimum for most models
            }
        ];

        let default_category = ModelType::Upscaling;
        let default_model = models.iter()
            .find(|m| m.model_type == default_category)
            .cloned();

        (
            Self {
                input_path: None,
                input_type: InputType::None,
                available_models: models.clone(),
                selected_category: Some(default_category),
                selected_model: default_model,
                image_files: Vec::new(),
                selected_preview_file: None,
                before_image: None,
                after_image: None,
                process_results: Vec::new(),
                processing: false,
                status_message: "Select an image or folder to begin".to_string(),
                zoom_level: 1.0,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "Super-Resolution Upscaler".to_string()
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::BrowseVideo => {
                return Command::perform(
                    async {
                        rfd::AsyncFileDialog::new()
                            .add_filter("Videos", &["mp4", "avi", "mkv", "mov", "webm"])
                            .pick_file()
                            .await
                            .map(|f| f.path().to_path_buf())
                    },
                    Message::VideoSelected,
                );
            }
            
            Message::VideoSelected(path) => {
                if let Some(path) = path {
                    self.input_path = Some(path.clone());
                    self.input_type = InputType::Video;
                    self.status_message = format!("Video loaded: {}", path.display());
                    self.after_image = None;
                    self.process_results.clear();
                }
            }
            
            Message::ProcessVideo => {
                if self.processing || self.input_path.is_none() {
                    return Command::none();
                }
                
                let Some(model) = self.selected_model.clone() else {
                    self.status_message = "No model selected".to_string();
                    return Command::none();
                };
                
                let Some(video_path) = self.input_path.clone() else {
                    return Command::none();
                };
                
                self.processing = true;
                self.status_message = "Processing video...".to_string();
                
                return Command::perform(
                    process_video(video_path, model),
                    Message::VideoProcessComplete,
                );
            }
            
            Message::VideoProcessComplete(result) => {
                self.processing = false;
                
                match result {
                    Ok(output_path) => {
                        self.status_message = format!("Video saved to: {}", output_path);
                    }
                    Err(e) => {
                        self.status_message = format!("Error: {}", e);
                    }
                }
            }
            
            Message::CategorySelected(category) => {
                self.selected_category = Some(category.clone());
                // Select the first model of the new category
                self.selected_model = self.available_models.iter()
                    .find(|m| m.model_type == category)
                    .cloned();
            }
            
            Message::BrowseFile => {
                return Command::perform(
                    async {
                        rfd::AsyncFileDialog::new()
                            .add_filter("Images", &["jpg", "jpeg", "png", "bmp", "webp"])
                            .pick_file()
                            .await
                            .map(|f| f.path().to_path_buf())
                    },
                    Message::FileSelected,
                );
            }
            Message::BrowseFolder => {
                return Command::perform(
                    async {
                        rfd::AsyncFileDialog::new()
                            .pick_folder()
                            .await
                            .map(|f| f.path().to_path_buf())
                    },
                    Message::FolderSelected,
                );
            }
            Message::FileSelected(path) => {
                if let Some(path) = path {
                    self.input_path = Some(path.clone());
                    self.input_type = InputType::File;
                    self.image_files = vec![path.clone()];
                    self.selected_preview_file = path.file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_string());
                    self.after_image = None;
                    self.process_results.clear();
                    self.status_message = format!("Loaded: {}", path.display());
                    self.zoom_level = 1.0;
                    
                    return Command::perform(
                        async move { 
                            image::open(&path)
                                .map(|img| (img, path.clone()))
                                .map_err(|e| e.to_string())
                        },
                        Message::PreviewLoaded,
                    );
                }
            }
            Message::FolderSelected(path) => {
                if let Some(path) = path {
                    let extensions = ["jpg", "jpeg", "png", "bmp", "webp"];
                    let mut files = Vec::new();
                    
                    if let Ok(entries) = std::fs::read_dir(&path) {
                        for entry in entries.flatten() {
                            let entry_path = entry.path();
                            if entry_path.is_file() {
                                if let Some(ext) = entry_path.extension() {
                                    if let Some(ext_str) = ext.to_str() {
                                        if extensions.contains(&ext_str.to_lowercase().as_str()) {
                                            files.push(entry_path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    files.sort();
                    
                    if files.is_empty() {
                        self.status_message = "No images found in folder".to_string();
                    } else {
                        self.input_path = Some(path);
                        self.input_type = InputType::Folder;
                        self.selected_preview_file = files.first()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .map(|s| s.to_string());
                        self.image_files = files.clone();
                        self.after_image = None;
                        self.process_results.clear();
                        self.status_message = format!("Loaded {} images", self.image_files.len());
                        self.zoom_level = 1.0;
                        
                        if let Some(first) = files.first() {
                            let path = first.clone();
                            return Command::perform(
                                async move {
                                    image::open(&path)
                                        .map(|img| (img, path.clone()))
                                        .map_err(|e| e.to_string())
                                },
                                Message::PreviewLoaded,
                            );
                        }
                    }
                }
            }
            Message::ModelSelected(model) => {
                self.selected_model = Some(model);
            }
            Message::PreviewFileSelected(filename) => {
                self.selected_preview_file = Some(filename.clone());
                self.zoom_level = 1.0;
                
                if let Some(file_path) = self.image_files.iter()
                    .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(&filename)) {
                    let path = file_path.clone();
                    
                    return Command::perform(
                        async move {
                            image::open(&path)
                                .map(|img| (img, path.clone()))
                                .map_err(|e| e.to_string())
                        },
                        Message::PreviewLoaded,
                    );
                }
            }
            Message::PreviewLoaded(result) => {
                match result {
                    Ok((img, path)) => {
                        self.before_image = Some(Arc::new(img));
                        
                        if let Some(result) = self.process_results.iter()
                            .find(|r| r.input_path == path) {
                            if let Ok(after_img) = image::open(&result.output_path) {
                                self.after_image = Some(Arc::new(after_img));
                            }
                        }
                    }
                    Err(e) => {
                        self.status_message = format!("Error: {}", e);
                    }
                }
            }
            Message::Process => {
                if self.processing || self.image_files.is_empty() {
                    return Command::none();
                }
                
                let Some(model) = self.selected_model.clone() else {
                    self.status_message = "No model selected".to_string();
                    return Command::none();
                };
                
                self.processing = true;
                self.status_message = "Processing...".to_string();
                
                let files = self.image_files.clone();
                let output_dir = if self.input_type == InputType::Folder {
                    self.input_path.as_ref()
                        .map(|p| p.join("processed"))
                        .unwrap_or_else(|| PathBuf::from("./processed"))
                } else {
                    PathBuf::from("./processed")
                };
                
                return Command::perform(
                    process_images(files, model, output_dir),
                    Message::ProcessComplete,
                );
            }
            Message::ProcessComplete(result) => {
                self.processing = false;
                
                match result {
                    Ok(results) => {
                        self.process_results = results.clone();
                        self.status_message = format!("Completed {} image(s)", results.len());
                        
                        if let Some(filename) = &self.selected_preview_file {
                            if let Some(file_path) = self.image_files.iter()
                                .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(filename)) {
                                
                                if let Some(result) = results.iter().find(|r| &r.input_path == file_path) {
                                    if let Ok(after_img) = image::open(&result.output_path) {
                                        self.after_image = Some(Arc::new(after_img));
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        self.status_message = format!("Error: {}", e);
                    }
                }
            }
            Message::ZoomIn => {
                self.zoom_level = (self.zoom_level * 1.2).min(5.0);
            }
            Message::ZoomOut => {
                self.zoom_level = (self.zoom_level / 1.2).max(0.1);
            }
            Message::ResetZoom => {
                self.zoom_level = 1.0;
            }
        }
        
        Command::none()
    }

    fn view(&self) -> Element<Message> {
        let header = container(
            column![
                text("Super-Resolution Upscaler")
                    .size(16)
                    .font(HEADING_FONT)
                    .style(Color::WHITE),
                text("AI-powered upscaling, denoising & restoration")
                    .size(11)
                    .font(BODY_FONT)
                    .style(Color::from_rgba(1.0, 1.0, 1.0, 0.8)),
            ].spacing(4)
        )
        .width(Length::Fill)
        .padding([18, 26])
        .style(theme::Container::Custom(Box::new(GradientContainer)));

        let file_btn = button("Browse File").on_press(Message::BrowseFile).padding(10);
        let folder_btn = button("Browse Folder").on_press(Message::BrowseFolder).padding(10);
        
        let video_btn = button("Browse Video")
            .on_press(Message::BrowseVideo)
            .padding(10);
    
        let input_card = card_container(
            column![
                section_title("Input"),
                Space::with_height(8),
                row![
                    file_btn,
                    folder_btn,
                    video_btn,
                    text(self.input_path.as_ref()
                        .and_then(|p| p.to_str())
                        .unwrap_or("No file selected"))
                        .size(14)
                        .style(TEXT_SECONDARY)
                ]
                .spacing(10)
                .align_items(Alignment::Center),
            ].spacing(0)
        );

        // Category picker
        let categories = vec![
            ModelType::Upscaling,
            ModelType::Enhancement,
            ModelType::Denoising,
        ];
        
        let category_picker = pick_list(
            categories,
            self.selected_category.as_ref(),
            Message::CategorySelected,
        )
        .placeholder("Select category");

        // Filter models by selected category
        let filtered_models: Vec<ModelInfo> = if let Some(category) = &self.selected_category {
            self.available_models.iter()
                .filter(|m| &m.model_type == category)
                .cloned()
                .collect()
        } else {
            self.available_models.clone()
        };

        let model_picker = pick_list(
            filtered_models,
            self.selected_model.as_ref(),
            Message::ModelSelected,
        )
        .placeholder("Select model");

        let process_btn = if self.processing {
            button(text("Processing...").font(HEADING_FONT).size(14))
                .padding([8, 10])
                .style(theme::Button::Secondary)
        } else {
            let btn_text = if self.input_type == InputType::Video {
                "Process Video"
            } else {
                "Start Processing"
            };
            
            let message = if self.input_type == InputType::Video {
                Message::ProcessVideo
            } else {
                Message::Process
            };
            
            button(text(btn_text).font(HEADING_FONT).size(14))
                .on_press(message)
                .padding([8, 10])
                .style(theme::Button::Primary)
        };

        let mut settings_card_content = column![
            section_title("Settings"),
            Space::with_height(8),
            row![
                text("Category:").size(14).style(TEXT_SECONDARY).width(Length::Fixed(80.0)),
                category_picker
            ].spacing(10).align_items(Alignment::Center),
            Space::with_height(8),
            row![
                text("Model:").size(14).style(TEXT_SECONDARY).width(Length::Fixed(80.0)),
                model_picker
            ].spacing(10).align_items(Alignment::Center),
            Space::with_height(12),
            process_btn,
            Space::with_height(8),
            text(&self.status_message).size(12).style(TEXT_SECONDARY),
        ]
        .spacing(0);

        if self.input_type == InputType::Folder && !self.image_files.is_empty() {
            let filenames: Vec<String> = self.image_files.iter()
                .filter_map(|p| p.file_name())
                .filter_map(|n| n.to_str())
                .map(|s| s.to_string())
                .collect();
            
            if !filenames.is_empty() {
                let file_picker = pick_list(
                    filenames,
                    self.selected_preview_file.as_ref(),
                    Message::PreviewFileSelected,
                )
                .placeholder("Select file");
                
                settings_card_content = settings_card_content.push(Space::with_height(12));
                settings_card_content = settings_card_content.push(
                    row![
                        text("Preview:").size(14).style(TEXT_SECONDARY).width(Length::Fixed(80.0)),
                        file_picker
                    ]
                    .spacing(10)
                    .align_items(Alignment::Center)
                );
            }
        }

        let settings_card = card_container(settings_card_content);

        let zoom_controls = row![
            button(text("-").size(18).horizontal_alignment(iced::alignment::Horizontal::Center))
                .on_press(Message::ZoomOut)
                .padding([4, 12])
                .style(theme::Button::Secondary),
            text(format!("{:.0}%", self.zoom_level * 100.0))
                .size(14)
                .style(TEXT_SECONDARY),
            button(text("+").size(18).horizontal_alignment(iced::alignment::Horizontal::Center))
                .on_press(Message::ZoomIn)
                .padding([4, 12])
                .style(theme::Button::Secondary),
            button(text("Reset").size(14))
                .on_press(Message::ResetZoom)
                .padding([4, 12])
                .style(theme::Button::Text),
        ]
        .spacing(8)
        .align_items(Alignment::Center)
        .width(Length::FillPortion(1));

        let preview_card = if let Some(before_img) = &self.before_image {
            let (w, h) = before_img.dimensions();
            let display_w = (w as f32 * self.zoom_level) as u32;
            let display_h = (h as f32 * self.zoom_level) as u32;

            let before_rgba = before_img.to_rgba8();
            let before_handle = iced_image::Handle::from_pixels(
                w,
                h,
                before_rgba.into_raw()
            );

            let before_preview = scrollable(
                container(
                    iced_image::Image::new(before_handle.clone())
                        .width(Length::Fixed(display_w as f32))
                        .height(Length::Fixed(display_h as f32))
                )
                .center_x()
                .center_y()
            )
            .direction(Direction::Both {
                vertical: Properties::default(),
                horizontal: Properties::default(),
            })
            .width(Length::FillPortion(1))
            .height(Length::Fixed(400.0));

            let before_col = column![
                text("Before").size(16).font(HEADING_FONT).style(TEXT_COLOR),
                Space::with_height(8),
                before_preview,
                Space::with_height(8),
                text(format!("{}×{}", w, h)).size(12).style(TEXT_SECONDARY)
            ]
            .spacing(0)
            .align_items(Alignment::Center);

            let after_col = if let Some(after_img) = &self.after_image {
                let (w, h) = after_img.dimensions();
                let display_w = (w as f32 * self.zoom_level) as u32;
                let display_h = (h as f32 * self.zoom_level) as u32;

                let after_rgba = after_img.to_rgba8();
                let after_handle = iced_image::Handle::from_pixels(
                    w,
                    h,
                    after_rgba.into_raw()
                );

                let after_preview = scrollable(
                    container(
                        iced_image::Image::new(after_handle)
                            .width(Length::Fixed(display_w as f32))
                            .height(Length::Fixed(display_h as f32))
                    )
                    .center_x()
                    .center_y()
                )
                .direction(Direction::Both {
                    vertical: Properties::default(),
                    horizontal: Properties::default(),
                })
                .width(Length::FillPortion(1))
                .height(Length::Fixed(400.0));

                column![
                    text("After").size(16).font(HEADING_FONT).style(TEXT_COLOR),
                    Space::with_height(8),
                    after_preview,
                    Space::with_height(8),
                    text(format!("{}×{}", w, h)).size(12).style(TEXT_SECONDARY)
                ]
                .spacing(0)
                .align_items(Alignment::Center)
            } else {
                column![
                    text("After").size(16).font(HEADING_FONT).style(TEXT_COLOR),
                    Space::with_height(8),
                    container(text("Process to see result").style(TEXT_SECONDARY))
                        .width(Length::Fixed(500.0))
                        .height(Length::Fixed(400.0))
                        .center_x()
                        .center_y()
                ]
                .spacing(0)
                .align_items(Alignment::Center)
                .width(Length::FillPortion(1))
            };

            card_container(
                column![
                    row![
                        section_title("Preview"),
                        Space::with_width(Length::Fill),
                        zoom_controls,
                    ],
                    Space::with_height(16),
                    row![before_col, Space::with_width(20), after_col]
                        .align_items(Alignment::Start),
                ].spacing(0)
            )
        } else {
            card_container(
                column![
                    section_title("Preview"),
                    Space::with_height(16),
                    text("Select an image to preview").size(14).style(TEXT_SECONDARY)
                ].spacing(0)
            )
        };

        let content = scrollable(
            column![
                header,
                container(
                    column![
                        input_card,
                        settings_card,
                        preview_card,
                        Space::with_height(20),
                    ].spacing(16)
                )
                .width(Length::Fill)
                .center_x()
                .padding([6, 14, 6, 6])
            ].spacing(0)
        );

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(theme::Container::Custom(Box::new(BackgroundContainer)))
            .into()
    }

    fn theme(&self) -> Theme {
        Theme::Light
    }
}

fn section_title(title: &str) -> Element<'static, Message> {
    text(title)
        .size(14)
        .font(HEADING_FONT)
        .style(TEXT_COLOR)
        .into()
}

fn card_container<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    container(content)
        .width(Length::Fill)
        .padding(14)
        .style(theme::Container::Custom(Box::new(CardContainer)))
        .into()
}

struct BackgroundContainer;
impl container::StyleSheet for BackgroundContainer {
    type Style = Theme;
    
    fn appearance(&self, _style: &Self::Style) -> container::Appearance {
        container::Appearance {
            background: Some(Background::Color(BACKGROUND_COLOR)),
            ..Default::default()
        }
    }
}

struct CardContainer;
impl container::StyleSheet for CardContainer {
    type Style = Theme;
    
    fn appearance(&self, _style: &Self::Style) -> container::Appearance {
        container::Appearance {
            background: Some(Background::Color(CARD_COLOR)),
            border: iced::Border {
                color: Color::from_rgba(0.0, 0.0, 0.0, 0.08),
                width: 1.0,
                radius: 12.0.into(),
            },
            ..Default::default()
        }
    }
}

struct GradientContainer;
impl container::StyleSheet for GradientContainer {
    type Style = Theme;
    
    fn appearance(&self, _style: &Self::Style) -> container::Appearance {
        container::Appearance {
            background: Some(Background::Color(PRIMARY_COLOR)),
            ..Default::default()
        }
    }
}

// Add this logging function at the top level
fn log_message(message: &str) {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    let log_entry = format!("[{}] {}\n", timestamp, message);
    
    // Print to console
    println!("{}", log_entry.trim());
    
    // Write to log file
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("image_processor.log")
    {
        let _ = file.write_all(log_entry.as_bytes());
    }
}

fn log_error(message: &str) {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    let log_entry = format!("[{}] ERROR: {}\n", timestamp, message);
    
    eprintln!("{}", log_entry.trim());
    
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("image_processor.log")
    {
        let _ = file.write_all(log_entry.as_bytes());
    }
}

// FIXED: Correct normalization for different model types
fn preprocess_image_for_model(img: &DynamicImage, model: &ModelInfo) -> Result<Array4<f32>> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    
    let normalize_fn: Box<dyn Fn(u8) -> f32> = match model.input_norm {
        NormalizationRange::MinusOneOne => {
            log_message(&format!("Input normalization: [-1, 1] for model: {}", model.name));
            Box::new(|val: u8| (val as f32 / 127.5) - 1.0)
        }
        NormalizationRange::ZeroOne => {
            log_message(&format!("Input normalization: [0, 1] for model: {}", model.name));
            Box::new(|val: u8| val as f32 / 255.0)
        }
    };
    
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            tensor[[0, 0, y as usize, x as usize]] = normalize_fn(p[0]);
            tensor[[0, 1, y as usize, x as usize]] = normalize_fn(p[1]);
            tensor[[0, 2, y as usize, x as usize]] = normalize_fn(p[2]);
        }
    }
    
    Ok(tensor)
}

// Update postprocessing function:
fn postprocess_tensor_for_model(tensor: Array4<f32>, model: &ModelInfo) -> Result<DynamicImage> {
    let shape = tensor.shape();
    let (_, _, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let mut img = ImageBuffer::new(w as u32, h as u32);
    
    let denormalize_fn: Box<dyn Fn(f32) -> u8> = match model.output_norm {
        NormalizationRange::MinusOneOne => {
            log_message(&format!("Output denormalization: [-1, 1] for model: {}", model.name));
            Box::new(|val: f32| ((val + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
        }
        NormalizationRange::ZeroOne => {
            log_message(&format!("Output denormalization: [0, 1] for model: {}", model.name));
            Box::new(|val: f32| (val * 255.0).clamp(0.0, 255.0) as u8)
        }
    };
    
    for y in 0..h {
        for x in 0..w {
            let r = denormalize_fn(tensor[[0, 0, y, x]]);
            let g = denormalize_fn(tensor[[0, 1, y, x]]);
            let b = denormalize_fn(tensor[[0, 2, y, x]]);
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    Ok(DynamicImage::ImageRgb8(img))
}

// IMPROVED: Better error handling in process_single_image
fn process_single_image(
    input_path: &Path,
    model: &ModelInfo,
    output_dir: &Path,
) -> Result<ProcessResult> {
    log_message(&format!("=== Processing: {} ===", input_path.display()));
    log_message(&format!("Model: {} ({})", model.name, model.category));
    
    let start = std::time::Instant::now();
    
    let model_path = format!("./models/{}.onnx", model.name);
    if !Path::new(&model_path).exists() {
        log_message(&format!("Model not found locally, downloading: {}", model.name));
        download_model(&model.url, &model_path).map_err(|e| {
            log_error(&format!("Failed to download model: {}", e));
            e
        })?;
        log_message("Model downloaded successfully");
    }

    log_message("Creating ONNX session...");
    let mut session = Session::builder()
        .map_err(|e| {
            log_error(&format!("Failed to create session builder: {}", e));
            e
        })?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| {
            log_error(&format!("Failed to set optimization level: {}", e));
            e
        })?
        .with_execution_providers([
            ort::execution_providers::DirectMLExecutionProvider::default().build()
        ])
        .map_err(|e| {
            log_error(&format!("Failed to set execution provider: {}", e));
            e
        })?
        .commit_from_file(&model_path)
        .map_err(|e| {
            log_error(&format!("Failed to load model from {}: {}", model_path, e));
            e
        })?;

    log_message("Loading input image...");
    let img = image::open(input_path).map_err(|e| {
        log_error(&format!("Failed to open image: {}", e));
        e
    })?;
    
    let (orig_w, orig_h) = img.dimensions();
    log_message(&format!("Original image size: {}x{}", orig_w, orig_h));
    
    // Apply model-specific minimum dimension requirement
    let min_dim = model.min_dimension.unwrap_or(0);
    let max_dim = 512.max(min_dim); // Use at least the minimum, or 512
    
    let img = if orig_w > max_dim || orig_h > max_dim || orig_w < min_dim || orig_h < min_dim {
        // Need to resize - either too large or too small
        let target_dim = if orig_w < min_dim || orig_h < min_dim {
            // Too small - upscale to minimum
            let scale = (min_dim as f32 / orig_w.min(orig_h) as f32).max(1.0);
            let new_w = (orig_w as f32 * scale) as u32;
            let new_h = (orig_h as f32 * scale) as u32;
            log_message(&format!("Image too small, upscaling to {}x{} (scale: {:.2})", new_w, new_h, scale));
            (new_w, new_h)
        } else {
            // Too large - downscale to max_dim
            let scale = (max_dim as f32 / orig_w.max(orig_h) as f32).min(1.0);
            let new_w = (orig_w as f32 * scale) as u32;
            let new_h = (orig_h as f32 * scale) as u32;
            log_message(&format!("Resizing to {}x{} (scale: {:.2})", new_w, new_h, scale));
            (new_w, new_h)
        };
        
        img.resize_exact(target_dim.0, target_dim.1, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    let (padded_img, padded_dims, (pad_r, pad_b)) = if model.window_size > 1 {
        log_message(&format!("Padding to multiple of {}", model.window_size));
        pad_to_multiple(&img, model.window_size)?
    } else {
        (img.clone(), img.dimensions(), (0, 0))
    };

    log_message(&format!("Padded dimensions: {}x{} (pad_r: {}, pad_b: {})", 
        padded_dims.0, padded_dims.1, pad_r, pad_b));
    
    // Verify dimensions are valid
    if padded_dims.0 == 0 || padded_dims.1 == 0 {
        return Err(anyhow::anyhow!("Invalid padded dimensions: {}x{}", padded_dims.0, padded_dims.1));
    }

    log_message(&format!("Preprocessing image {}x{} for model: {}", 
        padded_img.dimensions().0, padded_img.dimensions().1, model.name));

    log_message("Preprocessing image...");
    let input_tensor = preprocess_image_for_model(&padded_img, model).map_err(|e| {
        log_error(&format!("Preprocessing failed: {}", e));
        e
    })?;
 
    log_message("Creating ONNX input value...");
    let input_value = Value::from_array(input_tensor).map_err(|e| {
        log_error(&format!("Failed to create input value: {}", e));
        e
    })?;
    
    let input_name = session.inputs[0].name.to_string();
    let output_name = session.outputs[0].name.to_string();
    log_message(&format!("Model input: '{}', output: '{}'", input_name, output_name));

    log_message("Running inference...");
    let outputs = session.run(ort::inputs![input_name.as_str() => input_value]).map_err(|e| {
        log_error(&format!("Inference failed: {}", e));
        e
    })?;

    log_message("Extracting output tensor...");
    let (output_shape, output_data) = outputs[output_name.as_str()]
        .try_extract_tensor::<f32>()
        .map_err(|e| {
            log_error(&format!("Failed to extract tensor: {}", e));
            e
        })?;
    
    let shape_vec = output_shape.as_ref().to_vec();
    log_message(&format!("Output tensor shape: {:?}", shape_vec));
    
    let output_array = Array4::from_shape_vec(
        (shape_vec[0] as usize, shape_vec[1] as usize, 
         shape_vec[2] as usize, shape_vec[3] as usize),
        output_data.to_vec()
    ).map_err(|e| {
        log_error(&format!("Failed to create output array: {}", e));
        e
    })?;

    log_message("Postprocessing tensor...");
    let mut final_img = postprocess_tensor_for_model(output_array, model).map_err(|e| {
        log_error(&format!("Postprocessing failed: {}", e));
        e
    })?;

    if pad_r > 0 || pad_b > 0 {
        let target_w = img.dimensions().0 * model.scale;
        let target_h = img.dimensions().1 * model.scale;
        log_message(&format!("Cropping padding: target {}x{}", target_w, target_h));
        final_img = final_img.crop_imm(0, 0, target_w, target_h);
    }
    
    let (out_w, out_h) = final_img.dimensions();
    log_message(&format!("Final output size: {}x{}", out_w, out_h));

    let output_filename = input_path.file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("output");
    
    let suffix = match model.model_type {
        ModelType::Upscaling | ModelType::Enhancement if model.scale > 1 => format!("_{}x", model.scale),
        ModelType::Denoising => "_denoised".to_string(),
        _ => "_enhanced".to_string(),
    };
    
    let output_path = output_dir.join(format!("{}{}.png", output_filename, suffix));
    
    log_message(&format!("Saving to: {}", output_path.display()));
    final_img.save(&output_path).map_err(|e| {
        log_error(&format!("Failed to save image: {}", e));
        e
    })?;

    let duration = start.elapsed().as_secs_f32();
    log_message(&format!("✓ Completed in {:.2}s", duration));

    Ok(ProcessResult {
        input_path: input_path.to_path_buf(),
        output_path,
        input_dims: (orig_w, orig_h),
        output_dims: (out_w, out_h),
        duration,
    })
}

// Update process_images to use better error handling
async fn process_images(
    files: Vec<PathBuf>,
    model: ModelInfo,
    output_dir: PathBuf,
) -> Result<Vec<ProcessResult>, String> {
    tokio::task::spawn_blocking(move || {
        log_message("Initializing ONNX Runtime...");
        ort::init().commit().map_err(|e| {
            log_error(&format!("Failed to initialize ONNX Runtime: {}", e));
            e.to_string()
        })?;
        
        std::fs::create_dir_all(&output_dir).map_err(|e| {
            log_error(&format!("Failed to create output directory: {}", e));
            e.to_string()
        })?;
        
        let mut results = Vec::new();
        let total = files.len();
        
        for (idx, file_path) in files.iter().enumerate() {
            log_message(&format!("\n>>> Processing {}/{}: {}", idx + 1, total, file_path.display()));
            
            match process_single_image(&file_path, &model, &output_dir) {
                Ok(result) => {
                    log_message(&format!("✓ Success: {} -> {}", 
                        file_path.file_name().unwrap_or_default().to_string_lossy(),
                        result.output_path.file_name().unwrap_or_default().to_string_lossy()));
                    results.push(result);
                },
                Err(e) => {
                    log_error(&format!("✗ Failed to process {}: {}", file_path.display(), e));
                    // Continue processing other images instead of stopping
                }
            }
        }
        
        log_message(&format!("\n=== Batch Complete: {}/{} successful ===", results.len(), total));
        Ok(results)
    })
    .await
    .map_err(|e| {
        log_error(&format!("Task join error: {}", e));
        e.to_string()
    })?
}

fn pad_to_multiple(img: &DynamicImage, multiple: u32) -> Result<(DynamicImage, (u32, u32), (u32, u32))> {
    let (w, h) = img.dimensions();
    let pad_w = ((w + multiple - 1) / multiple) * multiple;
    let pad_h = ((h + multiple - 1) / multiple) * multiple;
    let pad_r = pad_w - w;
    let pad_b = pad_h - h;
    
    if pad_r == 0 && pad_b == 0 {
        return Ok((img.clone(), (w, h), (0, 0)));
    }
    
    let mut padded = ImageBuffer::new(pad_w, pad_h);
    let rgb = img.to_rgb8();
    
    for y in 0..pad_h {
        for x in 0..pad_w {
            let src_x = if x < w { x } else { w - 1 - (x - w).min(w - 1) };
            let src_y = if y < h { y } else { h - 1 - (y - h).min(h - 1) };
            padded.put_pixel(x, y, *rgb.get_pixel(src_x, src_y));
        }
    }
    
    Ok((DynamicImage::ImageRgb8(padded), (pad_w, pad_h), (pad_r, pad_b)))
}

fn download_model(url: &str, path_str: &str) -> Result<()> {
    if url == "local" { return Ok(()); }
    
    let path = Path::new(path_str);
    
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .user_agent("image-enhancement-tool/1.0")
        .build()?;

    println!("Downloading from: {}", url);
    let mut resp = client.get(url).send()?;

    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {} for {}", resp.status(), url));
    }

    let tmp = path.with_extension("part");
    let mut out = fs::File::create(&tmp)?;

    io::copy(&mut resp, &mut out)?;

    fs::rename(&tmp, path)?;
    
    println!("Model saved to: {}", path.display());

    Ok(())
}

fn preprocess_image(img: &DynamicImage) -> Result<Array4<f32>> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            tensor[[0, 0, y as usize, x as usize]] = p[0] as f32 / 255.0;
            tensor[[0, 1, y as usize, x as usize]] = p[1] as f32 / 255.0;
            tensor[[0, 2, y as usize, x as usize]] = p[2] as f32 / 255.0;
        }
    }
    
    Ok(tensor)
}

fn postprocess_tensor(tensor: Array4<f32>) -> Result<DynamicImage> {
    let shape = tensor.shape();
    let (_, _, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let mut img = ImageBuffer::new(w as u32, h as u32);
    
    for y in 0..h {
        for x in 0..w {
            let r = (tensor[[0, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[[0, 1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (tensor[[0, 2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    Ok(DynamicImage::ImageRgb8(img))
}

async fn process_video(
    video_path: PathBuf,
    model: ModelInfo,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        process_video_blocking(&video_path, &model)
    })
    .await
    .map_err(|e| e.to_string())?
}

fn check_codec_available(codec_name: &str) -> bool {
    ProcessCommand::new("ffmpeg")
        .args(&["-codecs"])
        .output()
        .map(|output| {
            let codecs_list = String::from_utf8_lossy(&output.stdout);
            codecs_list.contains(codec_name)
        })
        .unwrap_or(false)
}

fn process_video_blocking(
    video_path: &Path,
    model: &ModelInfo,
) -> Result<String, String> {
    // Create temporary directories
    let temp_frames = PathBuf::from("./temp_frames");
    let temp_upscaled = PathBuf::from("./temp_upscaled");
    
    std::fs::create_dir_all(&temp_frames).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&temp_upscaled).map_err(|e| e.to_string())?;
    
	// Configure Rayon thread pool for GPU processing
    // For GPU-based inference, fewer threads often work better
    // This uses 1/2 of CPU cores, or minimum of 2, max of 8
    let num_cpus = num_cpus::get();
    let optimal_threads = (num_cpus / 2).max(2).min(8);
    
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .ok(); // Ignore error if already initialized
    
    println!("Using {} parallel threads for video processing", optimal_threads);
    println!("Extracting frames from video...");
    
    // Extract frames using ffmpeg
    let extract_status = ProcessCommand::new("ffmpeg")
        .args(&[
            "-i", video_path.to_str().unwrap(),
            "-qscale:v", "1",
            "-qmin", "1",
            "-qmax", "1",
            &format!("{}/frame_%06d.png", temp_frames.display())
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("Failed to run ffmpeg: {}. Make sure ffmpeg is installed.", e))?;
    
    if !extract_status.success() {
        return Err("Failed to extract frames from video".to_string());
    }
    
    // Get list of extracted frames
    let mut frame_files: Vec<PathBuf> = std::fs::read_dir(&temp_frames)
        .map_err(|e| e.to_string())?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    
    frame_files.sort();
    
    if frame_files.is_empty() {
        return Err("No frames extracted from video".to_string());
    }
    
    println!("Processing {} frames in parallel...", frame_files.len());
    
    // Initialize ONNX Runtime
    ort::init().commit().map_err(|e| e.to_string())?;
    
    // Use atomic counter for progress tracking across threads
    use std::sync::atomic::{AtomicUsize, Ordering};
    let processed = AtomicUsize::new(0);
    let total = frame_files.len();
    
	// Process frames IN PARALLEL using rayon
    frame_files.par_iter().for_each(|frame_path| {
        match process_single_image(frame_path, model, &temp_upscaled) {
            Ok(_) => {
                let count = processed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 10 == 0 || count == total {
                    println!("Processing frame {}/{}...", count, total);
                }
            },
            Err(e) => eprintln!("Error processing frame: {}", e),
        }
    });
    
    println!("Reassembling video...");
    
    // Get video properties for output
    let output_path = video_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!(
            "{}_upscaled.mp4",
            video_path.file_stem().and_then(|s| s.to_str()).unwrap_or("output")
        ));
    
    // Get original FPS - handle fractional framerates properly
    let fps_output = ProcessCommand::new("ffprobe")
        .args(&[
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path.to_str().unwrap()
        ])
        .output()
        .map_err(|e| format!("Failed to get FPS: {}. Make sure ffprobe is installed.", e))?;
    
    let fps_str = String::from_utf8_lossy(&fps_output.stdout).trim().to_string();
    
    // Convert fractional FPS (e.g., "30000/1001") to decimal or use as-is
    let fps = if fps_str.is_empty() { 
        "30".to_string() 
    } else if fps_str.contains('/') {
        // Try to convert fraction to decimal for better compatibility
        let parts: Vec<&str> = fps_str.split('/').collect();
        if parts.len() == 2 {
            if let (Ok(num), Ok(den)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                format!("{:.3}", num / den)
            } else {
                fps_str
            }
        } else {
            fps_str
        }
    } else {
        fps_str
    };
    
    println!("Video framerate: {} fps", fps);
    
    // Determine output suffix based on model type
    let suffix = match model.model_type {
        ModelType::Upscaling | ModelType::Enhancement if model.scale > 1 => format!("_{}x", model.scale),
        ModelType::Denoising => "_denoised".to_string(),
        _ => "_enhanced".to_string(),
    };
    
    // Check if audio stream exists
    let has_audio = ProcessCommand::new("ffprobe")
        .args(&[
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path.to_str().unwrap()
        ])
        .output()
        .map(|out| !out.stdout.is_empty())
        .unwrap_or(false);
    
    println!("Audio track detected: {}", has_audio);
    
    // Build ffmpeg command with better encoding settings
    let mut ffmpeg_args = vec![
        "-y".to_string(), // Overwrite output file
        "-framerate".to_string(), fps.clone(),
        "-i".to_string(), format!("{}/frame_%06d{}.png", temp_upscaled.display(), suffix),
        "-i".to_string(), video_path.to_str().unwrap().to_string(), // Always add original video
    ];
    
    // Map video from processed frames
    ffmpeg_args.extend([
        "-map".to_string(), "0:v:0".to_string(),
    ]);
    
    // Map audio from original video (use ? to make it optional if no audio exists)
    if has_audio {
        // Check which audio encoder is available
        let audio_encoder = if check_codec_available("aac") {
            "aac"
        } else if check_codec_available("libmp3lame") {
            "libmp3lame"
        } else {
            "copy" // Fallback to copying the original audio stream
        };
        
        println!("Using audio codec: {}", audio_encoder);
        
        ffmpeg_args.extend([
            "-map".to_string(), "1:a:0".to_string(),
            "-c:a".to_string(), audio_encoder.to_string(),
        ]);
        
        // Only add quality settings if we're encoding (not copying)
        if audio_encoder != "copy" {
            ffmpeg_args.extend([
                "-b:a".to_string(), "192k".to_string(),
            ]);
        }
    } else {
        println!("No audio track found in source video - creating video-only output");
    }
    
    // Video encoding settings with codec detection
    let video_encoder = if check_codec_available("libx264") {
        "libx264"
    } else if check_codec_available("h264") {
        "h264"
    } else {
        "mpeg4" // Universal fallback
    };
    
    println!("Using video codec: {}", video_encoder);
    
    ffmpeg_args.extend([
        "-c:v".to_string(), video_encoder.to_string(),
    ]);
    
    // Only add x264-specific settings if using libx264
    if video_encoder == "libx264" {
        ffmpeg_args.extend([
            "-preset".to_string(), "medium".to_string(),
            "-crf".to_string(), "18".to_string(),
        ]);
    } else {
        // Generic quality settings for other codecs
        ffmpeg_args.extend([
            "-q:v".to_string(), "2".to_string(), // High quality
        ]);
    }
    
    ffmpeg_args.extend([
        "-pix_fmt".to_string(), "yuv420p".to_string(), // CRITICAL: Ensures compatibility
        "-movflags".to_string(), "+faststart".to_string(), // Better for streaming/playback
        "-r".to_string(), fps,
        output_path.to_str().unwrap().to_string(),
    ]);
    
    println!("Running ffmpeg with args: {:?}", ffmpeg_args);
    
    // Run ffmpeg and CAPTURE stderr for debugging
    let reassemble_output = ProcessCommand::new("ffmpeg")
        .args(&ffmpeg_args)
        .output()
        .map_err(|e| format!("Failed to run ffmpeg: {}. Make sure ffmpeg is installed.", e))?;
    
    if !reassemble_output.status.success() {
        let stderr = String::from_utf8_lossy(&reassemble_output.stderr);
        eprintln!("FFmpeg error output:\n{}", stderr);
        return Err(format!("Failed to reassemble video. FFmpeg error:\n{}", stderr));
    }
    
    println!("Video reassembly complete!");
    
    // Cleanup temporary files
    let _ = std::fs::remove_dir_all(&temp_frames);
    let _ = std::fs::remove_dir_all(&temp_upscaled);
    
    Ok(output_path.to_string_lossy().to_string())
}