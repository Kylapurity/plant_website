import React, { useState, useContext, useRef } from 'react';
import { AuthContext } from '../AuthContext';

// Mock data for plant disease trends
const trendData = [
  { id: 1, disease: 'Leaf Blight', occurrences: 235, trend: 'increasing', severity: 'high' },
  { id: 2, disease: 'Powdery Mildew', occurrences: 189, trend: 'stable', severity: 'medium' },
  { id: 3, disease: 'Root Rot', occurrences: 156, trend: 'decreasing', severity: 'high' },
  { id: 4, disease: 'Bacterial Spot', occurrences: 120, trend: 'increasing', severity: 'medium' },
  { id: 5, disease: 'Rust', occurrences: 98, trend: 'stable', severity: 'low' },
];

// Helper function to get recommendations based on disease name
const getRecommendations = (diseaseName) => {
  // Default recommendations
  const defaultRecommendations = [
    'Remove and destroy infected leaves',
    'Ensure proper air circulation around plants',
    'Avoid overhead watering'
  ];
  
  // Disease-specific recommendations
  const diseaseRecommendations = {
    'Apple___Apple_scab': [
      'Remove and destroy fallen leaves',
      'Apply fungicide during the growing season',
      'Prune trees to improve air circulation',
      'Consider resistant apple varieties for future plantings'
    ],
    'Tomato___Early_blight': [
      'Remove and destroy infected leaves',
      'Apply copper-based fungicide',
      'Mulch around plants to prevent soil splash',
      'Rotate crops and avoid planting tomatoes in the same spot'
    ],
    'Grape___Black_rot': [
      'Remove mummified berries and infected leaves',
      'Apply fungicide before bloom and after',
      'Ensure good canopy management and air circulation',
      'Use appropriate fungicide rotation to prevent resistance'
    ],
    // Add more disease-specific recommendations as needed
  };
  
  // Return disease-specific recommendations if available, otherwise default
  return diseaseRecommendations[diseaseName] || defaultRecommendations;
};

const Dashboard = () => {
  const { currentUser, logout } = useContext(AuthContext);
  const [activeTab, setActiveTab] = useState('trends');
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  // Dynamic history state instead of mock data
  const [scanHistory, setScanHistory] = useState([]);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'increasing':
        return '↑';
      case 'decreasing':
        return '↓';
      default:
        return '→';
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'increasing':
        return 'text-red-500';
      case 'decreasing':
        return 'text-green-500';
      default:
        return 'text-yellow-500';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-green-100 text-green-800';
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisResult(null); // Clear previous results
      setError(null); // Clear any previous errors
    }
  };

  const handleAnalyzeClick = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Create FormData to send the image
      const formData = new FormData();
      formData.append('file', selectedImage);
      
      // Call the FastAPI backend
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Format the disease name for display (remove underscores and class prefixes)
      const rawDiseaseName = data.prediction;
      let displayDiseaseName = rawDiseaseName;
      
      // Remove plant name prefix and underscores for display
      if (rawDiseaseName.includes('___')) {
        const parts = rawDiseaseName.split('___');
        displayDiseaseName = parts[1].replace(/_/g, ' ');
        
        // Handle 'healthy' case
        if (displayDiseaseName === 'healthy') {
          displayDiseaseName = 'Healthy - No Disease Detected';
        }
      }
      
      // Get confidence (this would come from your model in the real implementation)
      // For now, we'll use a random value between 85-99
      const confidence = Math.floor(Math.random() * 15) + 85;
      
      // Create result object
      const result = {
        disease: displayDiseaseName,
        rawDiseaseName: rawDiseaseName, // Keep the original for reference
        confidence: confidence,
        recommendations: getRecommendations(rawDiseaseName)
      };
      
      setAnalysisResult(result);
    } catch (err) {
      console.error('Error during disease prediction:', err);
      setError('Failed to analyze the image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSaveToHistory = () => {
    if (!analysisResult || !previewUrl) return;

    // Create a new history item
    const newHistoryItem = {
      id: Date.now(), // Use timestamp as ID
      imageUrl: previewUrl,
      date: new Date().toISOString().split('T')[0],
      disease: analysisResult.disease,
      rawDiseaseName: analysisResult.rawDiseaseName,
      confidence: analysisResult.confidence,
      recommendations: analysisResult.recommendations
    };

    // Add to history (add at beginning of array)
    setScanHistory([newHistoryItem, ...scanHistory]);
    
    // Show confirmation message
    alert('Scan saved to your history!');
    
    // Clear the current analysis
    setSelectedImage(null);
    setPreviewUrl('');
    setAnalysisResult(null);
  };

  const viewHistoryDetails = (item) => {
    setSelectedHistoryItem(item);
  };

  const closeHistoryDetails = () => {
    setSelectedHistoryItem(null);
  };

  const handleNewScan = () => {
    setSelectedImage(null);
    setPreviewUrl('');
    setAnalysisResult(null);
    setError(null);
    setActiveTab('upload');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-green-600">Plant Disease Detector</h1>
              </div>
            </div>
            <div className="flex items-center">
              <span className="mr-4 text-gray-600">Hello, {currentUser.name}</span>
              <button
                onClick={logout}
                className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="mb-4 border-b border-gray-200">
            <ul className="flex flex-wrap -mb-px">
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${
                    activeTab === 'trends'
                      ? 'text-green-600 border-b-2 border-green-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  onClick={() => setActiveTab('trends')}
                >
                  Disease Trends
                </button>
              </li>
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${
                    activeTab === 'upload'
                      ? 'text-green-600 border-b-2 border-green-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  onClick={() => setActiveTab('upload')}
                >
                  Upload Image and Predict Disease
                </button>
              </li>
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${
                    activeTab === 'history'
                      ? 'text-green-600 border-b-2 border-green-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  onClick={() => setActiveTab('history')}
                >
                  My History
                </button>
              </li>
            </ul>
          </div>

          {activeTab === 'trends' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Current Plant Disease Trends</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Disease
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Occurrences
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Trend
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Severity
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {trendData.map((item) => (
                      <tr key={item.id}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">{item.disease}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-500">{item.occurrences}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`text-sm ${getTrendColor(item.trend)}`}>
                            {item.trend} {getTrendIcon(item.trend)}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getSeverityColor(
                              item.severity
                            )}`}
                          >
                            {item.severity}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'upload' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Upload Plant Image for Disease Detection</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                <div className="mb-6">
                  <p className="text-gray-600 mb-4">
                    Upload a clear image of the plant leaf or affected area. Our AI will analyze it and identify any
                    potential diseases.
                  </p>
                  
                  {!previewUrl ? (
                    <div className="flex flex-col items-center">
                      <div className="flex items-center justify-center w-full mb-4">
                        <label
                          htmlFor="dropzone-file"
                          className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                        >
                          <div className="flex flex-col items-center justify-center pt-5 pb-6">
                            <svg
                              className="w-10 h-10 mb-3 text-gray-400"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                              ></path>
                            </svg>
                            <p className="mb-2 text-sm text-gray-500">
                              <span className="font-semibold">Click to upload</span> or drag and drop
                            </p>
                            <p className="text-xs text-gray-500">PNG, JPG, or JPEG (MAX. 10MB)</p>
                          </div>
                          <input 
                            id="dropzone-file" 
                            type="file" 
                            className="hidden" 
                            accept="image/*"
                            ref={fileInputRef}
                            onChange={handleImageChange}
                          />
                        </label>
                      </div>
                      
                      {/* New Predict Button */}
                      <button
                        type="button"
                        className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded focus:outline-none focus:shadow-outline"
                        onClick={() => {
                          // If no image is selected, prompt to select one
                          if (!selectedImage) {
                            alert("Please upload an image first");
                            fileInputRef.current.click();
                          } else {
                            handleAnalyzeClick();
                          }
                        }}
                      >
                        Predict Plant Disease
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <div className="relative mb-4 w-full max-w-lg">
                        <img 
                          src={previewUrl} 
                          alt="Plant preview" 
                          className="rounded-lg shadow-md object-contain max-h-64 mx-auto"
                        />
                        <button
                          onClick={() => {
                            setSelectedImage(null);
                            setPreviewUrl('');
                            setAnalysisResult(null);
                            setError(null);
                          }}
                          className="absolute top-2 right-2 bg-red-500 hover:bg-red-700 text-white rounded-full p-1"
                        >
                          <svg 
                            className="w-5 h-5" 
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24" 
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path 
                              strokeLinecap="round" 
                              strokeLinejoin="round" 
                              strokeWidth="2" 
                              d="M6 18L18 6M6 6l12 12"
                            />
                          </svg>
                        </button>
                      </div>
                      
                      {!analysisResult && !error && (
                        <button
                          type="button"
                          className={`bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded focus:outline-none focus:shadow-outline ${
                            isAnalyzing ? 'opacity-75 cursor-not-allowed' : ''
                          }`}
                          onClick={handleAnalyzeClick}
                          disabled={isAnalyzing}
                        >
                          {isAnalyzing ? 'Analyzing...' : 'Predict Disease'}
                        </button>
                      )}
                      
                      {error && (
                        <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
                          {error}
                          <button
                            type="button"
                            className="ml-4 bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-4 rounded focus:outline-none focus:shadow-outline"
                            onClick={() => setError(null)}
                          >
                            Try Again
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {analysisResult && (
                  <div className="mt-8 border-t pt-6">
                    <h3 className="text-xl font-bold mb-4">Analysis Results</h3>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="mb-4">
                        <span className="font-semibold">Detected Disease:</span>{' '}
                        <span className={analysisResult.disease.includes('Healthy') ? 'text-green-600' : 'text-red-600'}>
                          {analysisResult.disease}
                        </span>
                      </div>
                      <div className="mb-4">
                        <span className="font-semibold">Confidence:</span>{' '}
                        <span>{analysisResult.confidence}%</span>
                      </div>
                      <div className="mb-6">
                        <h4 className="font-semibold mb-2">Recommendations:</h4>
                        <ul className="list-disc pl-5">
                          {analysisResult.recommendations.map((rec, index) => (
                            <li key={index} className="text-gray-700 mb-1">{rec}</li>
                          ))}
                        </ul>
                      </div>
                      
                      <div className="flex justify-center space-x-4">
                        <button
                          type="button"
                          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded focus:outline-none focus:shadow-outline"
                          onClick={handleSaveToHistory}
                        >
                          Save to History
                        </button>
                        <button
                          type="button"
                          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded focus:outline-none focus:shadow-outline"
                          onClick={() => {
                            setSelectedImage(null);
                            setPreviewUrl('');
                            setAnalysisResult(null);
                            setError(null);
                          }}
                        >
                          Scan Another Plant
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'history' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Your Detection History</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                {scanHistory.length === 0 ? (
                  <div>
                    <p className="text-gray-600 mb-4 text-center">You haven't analyzed any plants yet.</p>
                    <div className="flex justify-center">
                      <button
                        type="button"
                        className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded focus:outline-none focus:shadow-outline"
                        onClick={handleNewScan}
                      >
                        Upload and Predict
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    {selectedHistoryItem ? (
                      <div className="bg-white p-4 rounded-lg">
                        <div className="flex justify-between items-start mb-4">
                          <h3 className="text-xl font-bold">{selectedHistoryItem.disease}</h3>
                          <button
                            onClick={closeHistoryDetails}
                            className="text-gray-500 hover:text-gray-700"
                          >
                            <svg 
                              className="w-6 h-6" 
                              fill="none" 
                              stroke="currentColor" 
                              viewBox="0 0 24 24" 
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path 
                                strokeLinecap="round" 
                                strokeLinejoin="round" 
                                strokeWidth="2" 
                                d="M6 18L18 6M6 6l12 12"
                              />
                            </svg>
                          </button>
                        </div>
                        
                        <div className="flex flex-col md:flex-row md:space-x-6">
                          <div className="md:w-1/3 mb-4 md:mb-0">
                            <img 
                              src={selectedHistoryItem.imageUrl} 
                              alt={selectedHistoryItem.disease} 
                              className="rounded-lg shadow-md w-full h-auto"
                            />
                            <p className="text-gray-500 mt-2 text-sm">
                              Scanned on {selectedHistoryItem.date}
                            </p>
                          </div>
                          
                          <div className="md:w-2/3">
                            <div className="mb-4">
                              <span className="font-semibold">Confidence:</span>{' '}
                              <span>{selectedHistoryItem.confidence}%</span>
                            </div>
                            <div>
                              <h4 className="font-semibold mb-2">Recommendations:</h4>
                              <ul className="list-disc pl-5">
                                {selectedHistoryItem.recommendations.map((rec, index) => (
                                  <li key={index} className="text-gray-700 mb-1">{rec}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                        
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded focus:outline-none focus:shadow-outline"
                            onClick={handleNewScan}
                          >
                            Scan New Plant
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {scanHistory.map((item) => (
                            <div 
                              key={item.id} 
                              className="bg-white overflow-hidden shadow-md rounded-lg hover:shadow-lg transition-shadow cursor-pointer"
                              onClick={() => viewHistoryDetails(item)}
                            >
                              <div className="relative h-48 bg-gray-200">
                                <img 
                                  src={item.imageUrl} 
                                  alt={item.disease} 
                                  className="w-full h-full object-cover"
                                />
                              </div>
                              <div className="p-4">
                                <p className="text-gray-500 text-sm">{item.date}</p>
                                <h3 className={`font-bold mt-1 ${item.disease.includes('Healthy') ? 'text-green-600' : 'text-red-600'}`}>
                                  {item.disease}
                                </h3>
                                <p className="text-gray-700 mt-1">Confidence: {item.confidence}%</p>
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded focus:outline-none focus:shadow-outline"
                            onClick={handleNewScan}
                          >
                            Scan New Plant
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;