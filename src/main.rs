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
    ModelSelected(ModelInfo),
    PreviewFileSelected(String),
    Process,
    ProcessComplete(Result<Vec<ProcessResult>, String>),
    PreviewLoaded(Result<(DynamicImage, PathBuf), String>),
    ZoomIn,
    ZoomOut,
    ResetZoom,
}

struct App {
    input_path: Option<PathBuf>,
    input_type: InputType,
    available_models: Vec<ModelInfo>,
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
}

#[derive(Debug, Clone, PartialEq)]
enum ModelType {
    Upscaling,
    Denoising,
    Enhancement,
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
            },
            ModelInfo {
                name: "swin2SR-classical-sr-x4-64".to_string(),
                url: "https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 8,
                description: "Clean images (4x)".to_string(),
                category: "Swin2SR".to_string(),
            },
            ModelInfo {
                name: "swin2SR-lightweight-x2-64".to_string(),
                url: "https://huggingface.co/Xenova/swin2SR-lightweight-x2-64/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 2,
                window_size: 8,
                description: "Lightweight (2x)".to_string(),
                category: "Swin2SR".to_string(),
            },
            ModelInfo {
                name: "swin2SR-compressed-sr-x4-48".to_string(),
                url: "https://huggingface.co/Xenova/swin2SR-compressed-sr-x4-48/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 8,
                description: "Compressed/JPEG (4x)".to_string(),
                category: "Swin2SR".to_string(),
            },
            ModelInfo {
                name: "2x_APISR_RRDB_GAN_generator".to_string(),
                url: "https://huggingface.co/Xenova/2x_APISR_RRDB_GAN_generator-onnx/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 2,
                window_size: 1,
                description: "APISR GAN (2x) Anime".to_string(),
                category: "APISR".to_string(),
            },
            ModelInfo {
                name: "4x_APISR_GRL_GAN_generator".to_string(),
                url: "https://huggingface.co/Xenova/4x_APISR_GRL_GAN_generator-onnx/resolve/main/onnx/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 1,
                description: "APISR GAN (4x) Anime".to_string(),
                category: "APISR".to_string(),
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
            },
            ModelInfo {
                name: "SwinIR-BSRGAN-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/SwinIR-BSRGAN-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 8,
                description: "Real degradations (4x)".to_string(),
                category: "SwinIR".to_string(),
            },
            ModelInfo {
                name: "BSRGAN-2x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/BSRGAN-2x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 2,
                window_size: 1,
                description: "Blind SR (2x)".to_string(),
                category: "BSRGAN".to_string(),
            },
            ModelInfo {
                name: "RealESRGAN-2x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/RealESRGAN-2x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 2,
                window_size: 1,
                description: "Real-world SR (2x)".to_string(),
                category: "RealESRGAN".to_string(),
            },
            ModelInfo {
                name: "RealESRGAN-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/RealESRGAN-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "Real-world SR (4x)".to_string(),
                category: "RealESRGAN".to_string(),
            },
            ModelInfo {
                name: "RealESR-General-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/RealESR-General-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "General purpose (4x)".to_string(),
                category: "RealESRGAN".to_string(),
            },
            ModelInfo {
                name: "Swin2SR-Classical-2x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/Swin2SR-Classical-2x/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 2,
                window_size: 8,
                description: "Classical SR (2x)".to_string(),
                category: "Swin2SR-TS".to_string(),
            },
            ModelInfo {
                name: "Swin2SR-Classical-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/Swin2SR-Classical-4x/model.onnx".to_string(),
                model_type: ModelType::Upscaling,
                scale: 4,
                window_size: 8,
                description: "Classical SR (4x)".to_string(),
                category: "Swin2SR-TS".to_string(),
            },
            ModelInfo {
                name: "UltraSharp-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/UltraSharp-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "Ultra sharp details (4x)".to_string(),
                category: "Custom".to_string(),
            },
            ModelInfo {
                name: "UltraMix-Smooth-4x".to_string(),
                url: "https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/UltraMix-Smooth-4x/model.onnx".to_string(),
                model_type: ModelType::Enhancement,
                scale: 4,
                window_size: 1,
                description: "Ultra smooth details (4x)".to_string(),
                category: "Custom".to_string(),
            },
        ];

        (
            Self {
                input_path: None,
                input_type: InputType::None,
                available_models: models.clone(),
                selected_model: models.first().cloned(),
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
        "Image Enhancement Tool - Upscaling & Restoration".to_string()
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
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
                text("Image Enhancement Tool")
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
        
        let input_card = card_container(
            column![
                section_title("Input"),
                Space::with_height(8),
                row![
                    file_btn,
                    folder_btn,
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

        let model_picker = pick_list(
            &self.available_models[..],
            self.selected_model.as_ref(),
            Message::ModelSelected,
        )
        .placeholder("Select model");

        let process_btn = if self.processing {
            button(text("Processing...").font(HEADING_FONT).size(14))
                .padding([8, 10])
                .style(theme::Button::Secondary)
        } else {
            button(text("Start Processing").font(HEADING_FONT).size(14))
                .on_press(Message::Process)
                .padding([8, 10])
                .style(theme::Button::Primary)
        };

        let mut settings_card_content = column![
            section_title("Settings"),
            Space::with_height(8),
            row![
                text("Model:").size(14).style(TEXT_SECONDARY),
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
                    row![text("Preview:").size(14).style(TEXT_SECONDARY), file_picker]
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

async fn process_images(
    files: Vec<PathBuf>,
    model: ModelInfo,
    output_dir: PathBuf,
) -> Result<Vec<ProcessResult>, String> {
    tokio::task::spawn_blocking(move || {
        ort::init().commit().map_err(|e| e.to_string())?;
        
        std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;
        
        let mut results = Vec::new();
        
        for file_path in files {
            match process_single_image(&file_path, &model, &output_dir) {
                Ok(result) => results.push(result),
                Err(e) => eprintln!("Error processing {}: {}", file_path.display(), e),
            }
        }
        
        Ok(results)
    })
    .await
    .map_err(|e| e.to_string())?
}

fn process_single_image(
    input_path: &Path,
    model: &ModelInfo,
    output_dir: &Path,
) -> Result<ProcessResult> {
    let start = std::time::Instant::now();
    
    let model_path = format!("./models/{}.onnx", model.name);
    if !Path::new(&model_path).exists() {
        println!("Downloading model: {}", model.name);
        download_model(&model.url, &model_path)?;
        println!("Model downloaded successfully");
    }

    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_execution_providers([
            ort::execution_providers::DirectMLExecutionProvider::default().build()
        ])?
        .commit_from_file(&model_path)?;

    let img = image::open(input_path)?;
    let (orig_w, orig_h) = img.dimensions();
    
    const MAX_DIM: u32 = 512;
    let img = if orig_w > MAX_DIM || orig_h > MAX_DIM {
        let scale = (MAX_DIM as f32 / orig_w.max(orig_h) as f32).min(1.0);
        img.resize_exact(
            (orig_w as f32 * scale) as u32,
            (orig_h as f32 * scale) as u32,
            image::imageops::FilterType::Lanczos3
        )
    } else {
        img
    };

    let (padded_img, _, (pad_r, pad_b)) = if model.window_size > 1 {
        pad_to_multiple(&img, model.window_size)?
    } else {
        (img.clone(), img.dimensions(), (0, 0))
    };

    let input_tensor = preprocess_image(&padded_img)?;
    let input_value = Value::from_array(input_tensor)?;
    let input_name = session.inputs[0].name.to_string();
    let output_name = session.outputs[0].name.to_string();

    let outputs = session.run(ort::inputs![input_name.as_str() => input_value])?;

    let (output_shape, output_data) = outputs[output_name.as_str()]
        .try_extract_tensor::<f32>()?;
    
    let shape_vec = output_shape.as_ref().to_vec();
    let output_array = Array4::from_shape_vec(
        (shape_vec[0] as usize, shape_vec[1] as usize, 
         shape_vec[2] as usize, shape_vec[3] as usize),
        output_data.to_vec()
    )?;

    let mut final_img = postprocess_tensor(output_array)?;
    
    if pad_r > 0 || pad_b > 0 {
        let target_w = img.dimensions().0 * model.scale;
        let target_h = img.dimensions().1 * model.scale;
        final_img = final_img.crop_imm(0, 0, target_w, target_h);
    }
    
    let (out_w, out_h) = final_img.dimensions();

    let output_filename = input_path.file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("output");
    
    let suffix = match model.model_type {
        ModelType::Upscaling | ModelType::Enhancement if model.scale > 1 => format!("_{}x", model.scale),
        ModelType::Denoising => "_denoised".to_string(),
        _ => "_enhanced".to_string(),
    };
    
    let output_path = output_dir.join(format!("{}{}.png", output_filename, suffix));
    
    final_img.save(&output_path)?;

    Ok(ProcessResult {
        input_path: input_path.to_path_buf(),
        output_path,
        input_dims: (orig_w, orig_h),
        output_dims: (out_w, out_h),
        duration: start.elapsed().as_secs_f32(),
    })
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