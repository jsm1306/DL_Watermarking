import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ==================== MODEL BUILDING FUNCTIONS ====================
def build_encoder(input_shape=(32, 32, 3)):
    """Encoder: Compresses watermark to feature representation"""
    inputs = layers.Input(shape=input_shape, name='watermark_input')
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(24, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(24, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    encoded = layers.Conv2D(48, 1, padding='same', activation='tanh', name='encoded_watermark')(x)
    
    return Model(inputs, encoded, name='Encoder')

def build_embedder(cover_shape=(128, 128, 3), watermark_shape=(32, 32, 48)):
    """Embedder: Embeds encoded watermark into cover image"""
    cover_input = layers.Input(shape=cover_shape, name='cover_input')
    watermark_input = layers.Input(shape=watermark_shape, name='encoded_watermark_input')
    
    w_up = layers.UpSampling2D(size=(4, 4))(watermark_input)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(cover_input)
    x = layers.BatchNormalization()(x)
    
    x = layers.Concatenate()([x, w_up])
    
    x = layers.Conv2D(128, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    watermarked = layers.Conv2D(3, 1, padding='same', activation='sigmoid', name='watermarked_image')(x)
    
    return Model([cover_input, watermark_input], watermarked, name='Embedder')

def build_decoder(input_shape=(128, 128, 3)):
    """Decoder: Extracts watermark from watermarked image"""
    inputs = layers.Input(shape=input_shape, name='watermarked_input')
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling2D(pool_size=(4, 4))(x)
    
    x = layers.Conv2D(24, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(24, 1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    extracted = layers.Conv2D(3, 1, padding='same', activation='sigmoid', name='extracted_watermark')(x)
    
    return Model(inputs, extracted, name='Decoder')

def build_watermarking_model():
    """Complete end-to-end watermarking system"""
    cover_input = layers.Input(shape=(128, 128, 3), name='cover')
    watermark_input = layers.Input(shape=(32, 32, 3), name='watermark')
    
    encoder = build_encoder()
    encoded_wm = encoder(watermark_input)
    
    embedder = build_embedder()
    watermarked_img = embedder([cover_input, encoded_wm])
    
    decoder = build_decoder()
    extracted_wm = decoder(watermarked_img)
    
    model = Model(
        inputs=[cover_input, watermark_input],
        outputs=[watermarked_img, extracted_wm],
        name='WatermarkingSystem'
    )
    
    return model, encoder, embedder, decoder

# ==================== LOAD MODELS ====================
print("Loading models...")
try:
    # Build model architecture first
    full_model, encoder, embedder, decoder = build_watermarking_model()
    
    # Load weights
    full_model.load_weights('watermarking_model.h5')
    encoder.load_weights('encoder_model.h5')
    embedder.load_weights('embedder_model.h5')
    decoder.load_weights('decoder_model.h5')
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure all .h5 weight files are in the same directory as this script.")

# ==================== HELPER FUNCTIONS ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size):
    """Load and preprocess image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def save_output_image(image_array, filename):
    """Save numpy array as image"""
    img = (image_array * 255).astype(np.uint8)
    img = Image.fromarray(img)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    img.save(output_path)
    return output_path

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = (image_array * 255).astype(np.uint8)
    img = Image.fromarray(img)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_watermark():
    try:
        # Check if files are present
        if 'cover_image' not in request.files or 'watermark_image' not in request.files:
            return jsonify({'error': 'Both cover and watermark images are required'}), 400
        
        cover_file = request.files['cover_image']
        watermark_file = request.files['watermark_image']
        
        # Validate files
        if cover_file.filename == '' or watermark_file.filename == '':
            return jsonify({'error': 'No selected files'}), 400
        
        if not (allowed_file(cover_file.filename) and allowed_file(watermark_file.filename)):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, or JPEG'}), 400
        
        # Save uploaded files
        cover_filename = secure_filename(cover_file.filename)
        watermark_filename = secure_filename(watermark_file.filename)
        
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cover_' + cover_filename)
        watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], 'watermark_' + watermark_filename)
        
        cover_file.save(cover_path)
        watermark_file.save(watermark_path)
        
        # Preprocess images
        cover_img = preprocess_image(cover_path, (128, 128))
        watermark_img = preprocess_image(watermark_path, (32, 32))
        
        # Add batch dimension
        cover_batch = np.expand_dims(cover_img, axis=0)
        watermark_batch = np.expand_dims(watermark_img, axis=0)
        
        # Process through model
        watermarked, extracted = full_model.predict([cover_batch, watermark_batch], verbose=0)
        
        # Remove batch dimension
        watermarked_img = watermarked[0]
        extracted_img = extracted[0]
        
        # Save output images
        save_output_image(cover_img, 'original_cover.png')
        save_output_image(watermark_img, 'original_watermark.png')
        save_output_image(watermarked_img, 'watermarked.png')
        save_output_image(extracted_img, 'extracted_watermark.png')
        
        # Convert to base64 for display
        result = {
            'original_cover': image_to_base64(cover_img),
            'original_watermark': image_to_base64(watermark_img),
            'watermarked': image_to_base64(watermarked_img),
            'extracted_watermark': image_to_base64(extracted_img)
        }
        
        # Calculate quality metrics
        mse_cover = np.mean((cover_img - watermarked_img) ** 2)
        psnr = 10 * np.log10(1.0 / mse_cover) if mse_cover > 0 else float('inf')
        
        mse_watermark = np.mean((watermark_img - extracted_img) ** 2)
        
        result['metrics'] = {
            'psnr': f"{psnr:.2f} dB",
            'mse_cover': f"{mse_cover:.6f}",
            'mse_watermark': f"{mse_watermark:.6f}"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed images"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)