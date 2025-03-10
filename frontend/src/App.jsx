import React, { useState, useCallback } from 'react';
import { Upload, Image as ImageIcon, Loader2 } from 'lucide-react';
import axios from "axios";

const API_URL = "http://localhost:8000"

function App() {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [imageId, setImageId] = useState(null);
  const [publicUrl, setPublicUrl] = useState(null);
  const [processedUrl, setProcessedUrl] = useState(null);

  const onDrop = useCallback( async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      file.preview = URL.createObjectURL(file);
      setFile(file);
      setProcessedImage(null);

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await axios.post(`${API_URL}/api/upload`, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        setImageId(response.data.image_id);
        setPublicUrl(response.data.public_url);
      } catch (err) {
        console.error(err);
      }
    }
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    onDrop(droppedFiles);
  };

  const handleFileInput = (e) => {
    if (e.target.files?.length) {
      onDrop(Array.from(e.target.files));
    }
  };

  const processImage = async () => {
    if (!file) return;

    setIsProcessing(true);
    try {
      const response = await axios.post(`${API_URL}/api/img-inference/?image_id=${imageId}`);

      setProcessedUrl(response.data.processed_url);
    } catch (err) {
      setError("Failed to process image.");
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
    setProcessedImage(file.preview);
    setIsProcessing(false);
  };

  const downloadImage = async () => {
  
    try {
      const response = await axios.get(API_URL + processedUrl, {
        responseType: "blob",
      });
  
      const url = window.URL.createObjectURL(new Blob([response.data]));
  
      const a = document.createElement("a");
      a.href = url;
      a.download = `${imageId}.jpg`; 
      document.body.appendChild(a);
      a.click();
  
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Download failed:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Image Super Resolution</h1>

        <div className="space-y-8">
          {/* Upload Section */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors
              ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-500'}
              ${!file ? 'cursor-pointer' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {!file ? (
              <div className="space-y-4">
                <div className="flex justify-center">
                  <Upload className="w-12 h-12 text-gray-400" />
                </div>
                <div>
                  <p className="text-xl">Drag and drop your image here</p>
                  <p className="text-gray-400">or</p>
                  <label className="inline-block px-4 py-2 bg-blue-600 rounded-lg cursor-pointer hover:bg-blue-700 transition-colors mt-2">
                    Choose File
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleFileInput}
                    />
                  </label>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <img
                  src={API_URL + publicUrl}
                  alt="Preview"
                  className="max-h-96 mx-auto rounded-lg"
                />
                <button
                  onClick={() => setFile(null)}
                  className="text-red-400 hover:text-red-300"
                >
                  Remove image
                </button>
              </div>
            )}
          </div>

          {/* Process Button */}
          {file && (
            <div className="flex justify-center">
              <button
                onClick={processImage}
                disabled={isProcessing}
                className={`px-6 py-3 rounded-lg flex items-center space-x-2 text-lg font-medium
                  ${isProcessing
                    ? 'bg-gray-600 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'}`}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <ImageIcon className="w-5 h-5" />
                    <span>Process Image</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Result Section */}
          {processedImage && (
            <div className="border border-gray-700 rounded-lg p-6 bg-gray-800/50">
              <h2 className="text-xl font-semibold mb-4">Enhanced Result</h2>
              <img
                src={API_URL + processedUrl}
                alt="Processed"
                className="max-h-96 mx-auto rounded-lg"
              />
              <button
                  onClick={downloadImage}
                  className="text-blue-400 hover:text-red-300"
                >
                  Download
                </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;