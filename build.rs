// build.rs - Place this in the root of your project (next to Cargo.toml)

#[cfg(windows)]
fn main() {
    use winres::WindowsResource;
    
    WindowsResource::new()
        .set_icon("icon.ico") // Optional: add an icon.ico file
        .set("ProductName", "Image Resizer")
        .set("FileDescription", "Resize images by size and dimensions")
        .set("LegalCopyright", "Copyright (C) 2024")
        .compile()
        .unwrap();
}

#[cfg(not(windows))]
fn main() {
    // Nothing to do on non-Windows platforms
}