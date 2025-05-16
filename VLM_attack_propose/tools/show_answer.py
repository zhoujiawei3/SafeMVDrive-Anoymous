from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import base64
from PIL import Image
import io

app = Flask(__name__)

# Ensure templates and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Create HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Model Output Visualizer</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto py-8 px-4">
        <h1 class="text-3xl font-bold mb-6">VLM Model Output Visualizer</h1>
        
        <!-- File Input Form -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-bold mb-4">Load JSON File</h2>
            <form id="fileForm" class="flex items-center">
                <input type="text" id="jsonPath" placeholder="Enter JSON file path" class="flex-1 p-2 border border-gray-300 rounded-lg mr-4">
                <button type="submit" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded">Load</button>
            </form>
        </div>
        
        <!-- Item Navigation -->
        <div id="itemNavigation" class="hidden bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-bold mb-4">Navigation</h2>
            <div class="flex items-center">
                <button id="prevBtn" class="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-l">Previous</button>
                <select id="itemIndex" class="p-2 border-t border-b border-gray-300"></select>
                <button id="nextBtn" class="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-r">Next</button>
            </div>
        </div>
        
        <!-- Content Section (Hidden initially) -->
        <div id="contentSection" class="hidden bg-white rounded-lg shadow-md p-6">
            <!-- Image Display -->
            <div class="mb-6">
                <h2 class="text-xl font-bold mb-2">Image</h2>
                <p id="imagePath" class="text-gray-600 mb-2"></p>
                <div class="flex justify-center">
                    <img id="visualImage" src="" alt="Visualization Image" class="max-w-full max-h-96 border border-gray-300 rounded-lg">
                </div>
            </div>
            
            <!-- Ground Truth and Model Answer -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                    <h2 class="text-xl font-bold mb-2">Ground Truth</h2>
                    <div id="groundTruth" class="bg-green-50 p-4 rounded-lg text-green-800 border-l-4 border-green-500"></div>
                </div>
                
                <div>
                    <h2 class="text-xl font-bold mb-2">Model Answer</h2>
                    <div id="modelAnswer" class="bg-blue-50 p-4 rounded-lg text-blue-800 border-l-4 border-blue-500"></div>
                </div>
            </div>
            
            <!-- Model Output -->
            <div class="mb-6">
                <h2 class="text-xl font-bold mb-2">Model Output</h2>
                <div id="modelOutput" class="bg-gray-50 p-4 rounded-lg text-gray-800 border-l-4 border-gray-300 whitespace-pre-wrap font-mono"></div>
            </div>
            
            <!-- Judge Result -->
            <div class="mb-6">
                <h2 class="text-xl font-bold mb-2">Judge Result</h2>
                <div id="judgeResult" class="p-4 rounded-lg border-l-4"></div>
            </div>
        </div>
        
        <!-- Progress Information -->
        <div id="progressInfo" class="hidden bg-white rounded-lg shadow-md p-6 mt-6">
            <div class="flex justify-between items-center">
                <div>
                    <span class="font-bold">Progress:</span> 
                    <span id="currentIndex">0</span>/<span id="totalItems">0</span>
                </div>
                <div class="w-2/3 bg-gray-200 rounded-full h-4">
                    <div id="progressBar" class="bg-blue-600 h-4 rounded-full" style="width: 0%"></div>
                </div>
                <div>
                    <span id="progressPercent">0%</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let jsonData = null;
        let currentItemIndex = -1;
        let jsonFilePath = '';
        
        // Debug function
        function debugLog(message) {
            console.log(`[DEBUG] ${message}`);
        }
        
        // File loading
        document.getElementById('fileForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            jsonFilePath = document.getElementById('jsonPath').value;
            if (!jsonFilePath) {
                alert('Please enter a JSON file path');
                return;
            }
            
            fetch('/load-json', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ path: jsonFilePath }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                // Save data to global variable
                jsonData = data.data;
                
                // Check if it's a list or a single object
                if (Array.isArray(jsonData)) {
                    // Initialize navigation
                    initializeNavigation(jsonData);
                    
                    // Setup progress bar
                    document.getElementById('progressInfo').classList.remove('hidden');
                    document.getElementById('totalItems').textContent = jsonData.length;
                    updateProgressBar(0, jsonData.length);
                    
                    // Load first item
                    loadItemByIndex(0);
                } else {
                    // If it's a single object, display content directly
                    document.getElementById('itemNavigation').classList.add('hidden');
                    document.getElementById('progressInfo').classList.add('hidden');
                    loadItemData(jsonData, data.image_data);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while loading the JSON file.');
            });
        });
        
        // Initialize navigation
        function initializeNavigation(items) {
            const itemNav = document.getElementById('itemNavigation');
            itemNav.classList.remove('hidden');
            
            // Fill dropdown
            const select = document.getElementById('itemIndex');
            select.innerHTML = '';
            
            items.forEach((item, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `Item ${index + 1}`;
                select.appendChild(option);
            });
            
            // Add dropdown event listener
            select.addEventListener('change', function() {
                const index = parseInt(this.value);
                loadItemByIndex(index);
            });
            
            // Add prev/next button event listeners
            document.getElementById('prevBtn').addEventListener('click', function() {
                if (currentItemIndex > 0) {
                    loadItemByIndex(currentItemIndex - 1);
                }
            });
            
            document.getElementById('nextBtn').addEventListener('click', function() {
                if (currentItemIndex < items.length - 1) {
                    loadItemByIndex(currentItemIndex + 1);
                }
            });
        }
        
        // Load item by index
        function loadItemByIndex(index) {
            if (!jsonData || !Array.isArray(jsonData) || index < 0 || index >= jsonData.length) {
                return;
            }
            
            currentItemIndex = index;
            
            // Update dropdown
            const select = document.getElementById('itemIndex');
            select.value = index;
            
            // Update progress
            updateProgressBar(index + 1, jsonData.length);
            
            // Get selected item
            const selectedItem = jsonData[index];
            
            // Debug info
            debugLog(`Loading item at index ${index}`);
            debugLog(`Image path: ${selectedItem.image_path}`);
            
            // Load image
            fetch('/get-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    path: selectedItem.image_path
                }),
            })
            .then(response => {
                debugLog(`Image response status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                debugLog(`Image data received: ${data.image_data ? 'Yes' : 'No'}`);
                loadItemData(selectedItem, data.image_data);
            })
            .catch(error => {
                console.error('Error loading image:', error);
                loadItemData(selectedItem, null);
            });
        }
        
        // Update progress bar
        function updateProgressBar(current, total) {
            document.getElementById('currentIndex').textContent = current;
            document.getElementById('totalItems').textContent = total;
            
            const percentage = Math.floor((current / total) * 100);
            document.getElementById('progressPercent').textContent = `${percentage}%`;
            document.getElementById('progressBar').style.width = `${percentage}%`;
        }
        
        // Load item data to display
        function loadItemData(item, imageData) {
            // Show content section
            document.getElementById('contentSection').classList.remove('hidden');
            
            debugLog(`Loading item data: ${JSON.stringify(item, null, 2).substring(0, 100)}...`);
            
            // Update UI
            document.getElementById('imagePath').textContent = item.image_path || 'No image path';
            document.getElementById('visualImage').src = imageData || '/static/placeholder.png';
            
            debugLog(`Image path: ${item.image_path}`);
            debugLog(`Image data available: ${imageData ? 'Yes' : 'No'}`);
            
            // Display ground truth
            const groundTruthElem = document.getElementById('groundTruth');
            groundTruthElem.innerHTML = '';
            
            if (item.ground_truth && Object.keys(item.ground_truth).length > 0) {
                for (const [key, value] of Object.entries(item.ground_truth)) {
                    groundTruthElem.innerHTML += `<div>Vehicle ID: ${key}, Value: ${value}</div>`;
                }
            } else {
                groundTruthElem.textContent = 'N/A';
            }
            
            // Display model answer
            document.getElementById('modelAnswer').textContent = item.model_answer || 'N/A';
            
            // Display model output
            const modelOutputElem = document.getElementById('modelOutput');
            modelOutputElem.textContent = item.model_output || 'N/A';
            
            // Format the model output to highlight answer section
            if (item.model_output) {
                // Find <answer> tags and highlight them
                const formattedOutput = item.model_output.replace(
                    /(<answer>)(.*?)(<\/answer>)/g, 
                    '<span class="bg-yellow-100 font-bold px-1 rounded">$1$2$3</span>'
                );
                modelOutputElem.innerHTML = formattedOutput;
            }
            
            // Display judge result with appropriate color
            const judgeElem = document.getElementById('judgeResult');
            const judgeValue = item.judge !== undefined ? item.judge : 'N/A';
            
            judgeElem.textContent = `${judgeValue}`;
            
            // Color coding for judge result
            if (judgeValue === 1) {
                judgeElem.className = 'p-4 rounded-lg border-l-4 bg-green-50 text-green-800 border-green-500';
            } else if (judgeValue === 0) {
                judgeElem.className = 'p-4 rounded-lg border-l-4 bg-red-50 text-red-800 border-red-500';
            } else {
                judgeElem.className = 'p-4 rounded-lg border-l-4 bg-gray-50 text-gray-800 border-gray-300';
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Don't trigger shortcuts in text areas or inputs
            if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
                return;
            }
            
            // Left arrow - previous
            if (e.key === 'ArrowLeft') {
                document.getElementById('prevBtn').click();
            }
            // Right arrow - next
            else if (e.key === 'ArrowRight') {
                document.getElementById('nextBtn').click();
            }
        });
    </script>
</body>
</html>
    ''')

# Create a placeholder image
placeholder_image = Image.new('RGB', (800, 600), color='lightgray')
draw = getattr(Image, 'Draw', None)
if draw:
    from PIL import ImageDraw
    d = ImageDraw.Draw(placeholder_image)
    d.text((400, 300), "No Image Available", fill="black")
placeholder_image.save('static/placeholder.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/load-json', methods=['POST'])
def load_json():
    data = request.json
    json_path = data.get('path')
    
    if not json_path:
        return jsonify({'error': 'No JSON path provided'})
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Check if JSON has a "results" key containing an array
        if "results" in json_data and isinstance(json_data["results"], list):
            json_data = json_data["results"]
            
        # If it's a list, don't load image yet, wait for user to select specific item
        if isinstance(json_data, list):
            return jsonify({
                'data': json_data,
                'image_data': None
            })
        
        # If it's a single object, try to load the image
        image_data = get_image_data(json_data.get('image_path'))
        
        return jsonify({
            'data': json_data,
            'image_data': image_data
        })
    
    except FileNotFoundError:
        return jsonify({'error': f'File not found: {json_path}'})
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON file'})
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/get-image', methods=['POST'])
def get_image():
    data = request.json
    image_path = data.get('path')
    
    if not image_path:
        return jsonify({'image_data': None})
    
    image_data = get_image_data(image_path)
    return jsonify({'image_data': image_data})

def get_image_data(image_path):
    """Try to load image and return base64 encoded data"""
    if not image_path:
        return None
        
    # Print debug info
    print(f"Loading image from: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Warning: Image path does not exist: {image_path}")
        return None
    
    try:
        with Image.open(image_path) as img:
            # Resize image to fit display
            img.thumbnail((800, 600))
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)