from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import base64
from PIL import Image
import io

app = Flask(__name__)

# templatesstatic
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# HTML
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Annotation Editor</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto py-8 px-4">
        <h1 class="text-3xl font-bold mb-6">JSON Annotation Editor</h1>
        
        <!-- File Input Form -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-bold mb-4">Load JSON File</h2>
            <form id="fileForm" class="flex items-center">
                <input type="text" id="jsonPath" placeholder="Enter JSON file path" class="flex-1 p-2 border border-gray-300 rounded-lg mr-4">
                <button type="submit" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded">Load</button>
            </form>
        </div>
        
        <!-- Item Navigation (For List JSON) -->
        <div id="itemNavigation" class="hidden bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-bold mb-4">Navigation</h2>
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <button id="prevBtn" class="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-l">Previous</button>
                    <select id="itemIndex" class="p-2 border-t border-b border-gray-300"></select>
                    <button id="nextBtn" class="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-r">Next</button>
                </div>
                <div>
                    <span class="font-bold">Sample Token:</span> <span id="sampleToken" class="font-mono bg-gray-100 px-2 py-1 rounded">-</span>
                </div>
            </div>
        </div>
        
        <!-- Editor Section (Hidden initially) -->
        <div id="editorSection" class="hidden bg-white rounded-lg shadow-md p-6">
            <!-- Image Display -->
            <div class="mb-6">
                <h2 class="text-xl font-bold mb-2">Image</h2>
                <p id="imagePath" class="text-gray-600 mb-2"></p>
                <div class="flex justify-center">
                    <img id="annotationImage" src="" alt="Annotation Image" class="max-w-full max-h-96 border border-gray-300 rounded-lg">
                </div>
            </div>
            
            <!-- Data Fields -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                    <h2 class="text-xl font-bold mb-2">Answer</h2>
                    <div id="answer" class="bg-purple-50 p-4 rounded-lg text-purple-800 border-l-4 border-purple-500"></div>
                </div>
                
                <div>
                    <h2 class="text-xl font-bold mb-2">Ground Truth</h2>
                    <div id="groundTruth" class="bg-green-50 p-4 rounded-lg text-green-800 border-l-4 border-green-500"></div>
                </div>
                
                <div>
                    <h2 class="text-xl font-bold mb-2">Collision Type</h2>
                    <div id="collisionType" class="bg-blue-50 p-4 rounded-lg text-blue-800 border-l-4 border-blue-500"></div>
                </div>
                
                <div>
                    <h2 class="text-xl font-bold mb-2">Ego Vehicle Speed</h2>
                    <div id="egoVehicleSpeed" class="bg-yellow-50 p-4 rounded-lg text-yellow-800 border-l-4 border-yellow-500"></div>
                </div>
            </div>
            
            <!-- Editable Full Response -->
            <div class="mb-6">
                <h2 class="text-xl font-bold mb-2">Full Response (Editable)</h2>
                <textarea id="fullResponse" class="w-full h-64 p-4 border border-gray-300 rounded-lg font-mono text-sm"></textarea>
            </div>
            
            <!-- Save Button -->
            <div class="flex justify-between">
                <div>
                    <label class="inline-flex items-center">
                        <input type="checkbox" id="autoAdvance" class="form-checkbox h-5 w-5 text-blue-600" checked>
                        <span class="ml-2 text-gray-700">Auto-advance to next item after saving</span>
                    </label>
                </div>
                <button id="saveBtn" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded">Save Changes</button>
            </div>
            
            <!-- Status Message -->
            <div id="statusMessage" class="hidden mt-4 p-4 rounded-lg"></div>
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
        // JSON
        let jsonData = null;
        let currentItemIndex = -1;
        let jsonFilePath = '';
        
        // 
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
                
                // 
                jsonData = data.data;
                
                // 
                if (Array.isArray(jsonData)) {
                    // 
                    initializeNavigation(jsonData);
                    
                    // 
                    document.getElementById('progressInfo').classList.remove('hidden');
                    document.getElementById('totalItems').textContent = jsonData.length;
                    updateProgressBar(0, jsonData.length);
                    
                    // 
                    loadItemByIndex(0);
                } else {
                    // 
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
        
        // 
        function initializeNavigation(items) {
            const itemNav = document.getElementById('itemNavigation');
            itemNav.classList.remove('hidden');
            
            // 
            const select = document.getElementById('itemIndex');
            select.innerHTML = '';
            
            items.forEach((item, index) => {
                const option = document.createElement('option');
                option.value = index;
                
                // 
                let itemText = `Item ${index + 1}`;
                option.textContent = itemText;
                select.appendChild(option);
            });
            
            // 
            select.addEventListener('change', function() {
                const index = parseInt(this.value);
                loadItemByIndex(index);
            });
            
            // /
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
        
        // 
        function loadItemByIndex(index) {
            if (!jsonData || !Array.isArray(jsonData) || index < 0 || index >= jsonData.length) {
                return;
            }
            
            currentItemIndex = index;
            
            // 
            const select = document.getElementById('itemIndex');
            select.value = index;
            
            // 
            updateProgressBar(index + 1, jsonData.length);
            
            //  sample token 
            const sampleToken = document.getElementById('sampleToken');
            sampleToken.textContent = jsonData[index].sample_token || 'N/A';
            
            // 
            const selectedItem = jsonData[index];
            
            // 
            fetch('/get-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    path: selectedItem.image_path
                }),
            })
            .then(response => response.json())
            .then(data => {
                loadItemData(selectedItem, data.image_data);
            })
            .catch(error => {
                console.error('Error loading image:', error);
                loadItemData(selectedItem, null);
            });
        }
        
        // 
        function updateProgressBar(current, total) {
            document.getElementById('currentIndex').textContent = current;
            document.getElementById('totalItems').textContent = total;
            
            const percentage = Math.floor((current / total) * 100);
            document.getElementById('progressPercent').textContent = `${percentage}%`;
            document.getElementById('progressBar').style.width = `${percentage}%`;
        }
        
        // 
        function loadItemData(item, imageData) {
            // 
            document.getElementById('editorSection').classList.remove('hidden');
            
            // UI
            document.getElementById('imagePath').textContent = item.image_path || 'No image path';
            document.getElementById('annotationImage').src = imageData || '/static/placeholder.png';
            document.getElementById('answer').textContent = item.answer || 'N/A';
            
            // ground truth
            const groundTruthElem = document.getElementById('groundTruth');
            groundTruthElem.innerHTML = '';
            
            if (item.ground_truth) {
                for (const [key, value] of Object.entries(item.ground_truth)) {
                    groundTruthElem.innerHTML += `<div>Vehicle ID: ${key}, Value: ${value}</div>`;
                }
            } else {
                groundTruthElem.textContent = 'N/A';
            }
            
            document.getElementById('collisionType').textContent = item.collision_type || 'N/A';
            
            //  ego vehicle speed
            document.getElementById('egoVehicleSpeed').textContent = item.ego_init_v !== undefined ? 
                `${item.ego_init_v} km/h` : 'N/A';
            
            document.getElementById('fullResponse').value = item.full_response || '';
        }
        
        // 
        document.getElementById('saveBtn').addEventListener('click', function() {
            const fullResponse = document.getElementById('fullResponse').value;
            
            // 
            const requestData = { 
                path: jsonFilePath,
                full_response: fullResponse
            };
            
            // 
            if (Array.isArray(jsonData) && currentItemIndex >= 0) {
                requestData.is_array = true;
                requestData.index = currentItemIndex;
            }
            
            // 
            const saveBtn = document.getElementById('saveBtn');
            const originalText = saveBtn.textContent;
            saveBtn.textContent = 'Saving...';
            saveBtn.disabled = true;
            
            fetch('/save-json', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            })
            .then(response => response.json())
            .then(data => {
                // 
                saveBtn.textContent = originalText;
                saveBtn.disabled = false;
                
                const statusMsg = document.getElementById('statusMessage');
                if (data.success) {
                    statusMsg.classList.remove('hidden', 'bg-red-100', 'text-red-700');
                    statusMsg.classList.add('bg-green-100', 'text-green-700');
                    statusMsg.textContent = 'Changes saved successfully to source file!';
                    
                    // 
                    if (Array.isArray(jsonData) && currentItemIndex >= 0) {
                        jsonData[currentItemIndex].full_response = fullResponse;
                        
                        // 
                        const autoAdvance = document.getElementById('autoAdvance').checked;
                        if (autoAdvance && currentItemIndex < jsonData.length - 1) {
                            setTimeout(() => {
                                loadItemByIndex(currentItemIndex + 1);
                            }, 500); // 
                        }
                    } else {
                        jsonData.full_response = fullResponse;
                    }
                } else {
                    statusMsg.classList.remove('hidden', 'bg-green-100', 'text-green-700');
                    statusMsg.classList.add('bg-red-100', 'text-red-700');
                    statusMsg.textContent = data.error || 'Failed to save changes to source file.';
                }
                
                // 5
                setTimeout(() => {
                    statusMsg.classList.add('hidden');
                }, 5000);
            })
            .catch(error => {
                // 
                saveBtn.textContent = originalText;
                saveBtn.disabled = false;
                
                console.error('Error:', error);
                alert('An error occurred while saving changes to the source file.');
            });
        });
        
        // 
        document.addEventListener('keydown', function(e) {
            // 
            if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
                return;
            }
            
            //  - 
            if (e.key === 'ArrowLeft') {
                document.getElementById('prevBtn').click();
            }
            //  - 
            else if (e.key === 'ArrowRight') {
                document.getElementById('nextBtn').click();
            }
            // Ctrl+S - 
            else if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                document.getElementById('saveBtn').click();
            }
        });
    </script>
</body>
</html>
    ''')

# 
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
        
        # 
        if isinstance(json_data, list):
            # JSON
            for i, item in enumerate(json_data):
                if not isinstance(item, dict):
                    return jsonify({'error': f'Item at index {i} is not a dictionary'})
                
                # sample_token
                if 'sample_token' not in item:
                    item['sample_token'] = f'item_{i}'
                
                # ego_init_v
                if 'ego_init_v' not in item:
                    item['ego_init_v'] = None
            
            return jsonify({
                'data': json_data,
                'image_data': None
            })
        
        # 
        # ego_init_v
        if 'ego_init_v' not in json_data:
            json_data['ego_init_v'] = None
            
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
    """base64"""
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        with Image.open(image_path) as img:
            # 
            img.thumbnail((800, 600))
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

@app.route('/save-json', methods=['POST'])
def save_json():
    data = request.json
    json_path = data.get('path')
    full_response = data.get('full_response')
    is_array = data.get('is_array', False)
    index = data.get('index', -1)
    
    if not json_path:
        return jsonify({'error': 'No JSON path provided'})
    
    try:
        # 
        if not os.path.exists(json_path):
            return jsonify({'error': f'File not found: {json_path}'})
        
        # JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        #  full_response
        if is_array and isinstance(json_data, list) and 0 <= index < len(json_data):
            json_data[index]['full_response'] = full_response
        elif isinstance(json_data, dict):
            json_data['full_response'] = full_response
        else:
            return jsonify({'error': 'Invalid data structure or index'})
        
        # 
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            # 
            f.flush()
            os.fsync(f.fileno())  # 
        
        return jsonify({'success': True})
    
    except IOError as e:
        return jsonify({'error': f'I/O error when saving file: {str(e)}'})
    except Exception as e:
        return jsonify({'error': f'Error saving file: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)