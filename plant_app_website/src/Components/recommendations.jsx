// recommendations.js
export const getRecommendations = (diseaseName) => {
    // Default recommendations for all diseases
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
      'Apple___Black_rot': [
        'Prune out dead or diseased branches',
        'Apply fungicide during the growing season',
        'Remove mummified fruits from the tree and ground',
        'Ensure proper spacing between trees for air circulation'
      ],
      'Apple___Cedar_apple_rust': [
        'Remove nearby cedar trees if possible',
        'Apply fungicide during the growing season',
        'Prune trees to improve air circulation',
        'Use resistant apple varieties'
      ],
      'Apple___healthy': [
        'Maintain good tree health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Prune trees to improve air circulation',
        'Apply preventive fungicide if necessary'
      ],
      'Blueberry___healthy': [
        'Maintain good soil pH (4.5-5.5)',
        'Regularly inspect for signs of disease',
        'Ensure proper irrigation and drainage',
        'Apply mulch to retain soil moisture'
      ],
      'Cherry_(including_sour)___Powdery_mildew': [
        'Apply fungicide during the growing season',
        'Prune trees to improve air circulation',
        'Remove and destroy infected leaves',
        'Avoid overhead watering'
      ],
      'Cherry_(including_sour)___healthy': [
        'Maintain good tree health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Prune trees to improve air circulation',
        'Apply preventive fungicide if necessary'
      ],
      'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant corn varieties'
      ],
      'Corn_(maize)___Common_rust_': [
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant corn varieties',
        'Ensure proper spacing for air circulation'
      ],
      'Corn_(maize)___Northern_Leaf_Blight': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant corn varieties'
      ],
      'Corn_(maize)___healthy': [
        'Maintain good soil fertility',
        'Regularly inspect for signs of disease',
        'Ensure proper irrigation and drainage',
        'Apply preventive fungicide if necessary'
      ],
      'Grape___Black_rot': [
        'Remove mummified berries and infected leaves',
        'Apply fungicide before bloom and after',
        'Ensure good canopy management and air circulation',
        'Use appropriate fungicide rotation to prevent resistance'
      ],
      'Grape___Esca_(Black_Measles)': [
        'Prune out infected wood and destroy it',
        'Apply fungicide during the dormant season',
        'Ensure proper vine spacing for air circulation',
        'Avoid wounding the vine during pruning'
      ],
      'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
        'Apply fungicide during the growing season',
        'Remove and destroy infected leaves',
        'Ensure good canopy management and air circulation',
        'Use resistant grape varieties'
      ],
      'Grape___healthy': [
        'Maintain good vine health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Prune vines to improve air circulation',
        'Apply preventive fungicide if necessary'
      ],
      'Orange___Haunglongbing_(Citrus_greening)': [
        'Remove and destroy infected trees',
        'Control psyllid populations with insecticides',
        'Use disease-free planting material',
        'Regularly inspect for signs of disease'
      ],
      'Peach___Bacterial_spot': [
        'Apply copper-based bactericide during the growing season',
        'Prune trees to improve air circulation',
        'Remove and destroy infected leaves and fruit',
        'Use resistant peach varieties'
      ],
      'Peach___healthy': [
        'Maintain good tree health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Prune trees to improve air circulation',
        'Apply preventive fungicide if necessary'
      ],
      'Pepper,_bell___Bacterial_spot': [
        'Apply copper-based bactericide during the growing season',
        'Remove and destroy infected plants',
        'Avoid overhead watering',
        'Use disease-free seeds and transplants'
      ],
      'Pepper,_bell___healthy': [
        'Maintain good plant health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Ensure proper spacing for air circulation',
        'Apply preventive fungicide if necessary'
      ],
      'Potato___Early_blight': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant potato varieties'
      ],
      'Potato___Late_blight': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant potato varieties'
      ],
      'Potato___healthy': [
        'Maintain good soil fertility',
        'Regularly inspect for signs of disease',
        'Ensure proper irrigation and drainage',
        'Apply preventive fungicide if necessary'
      ],
      'Raspberry___healthy': [
        'Maintain good plant health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Prune plants to improve air circulation',
        'Apply preventive fungicide if necessary'
      ],
      'Soybean___healthy': [
        'Maintain good soil fertility',
        'Regularly inspect for signs of disease',
        'Ensure proper irrigation and drainage',
        'Apply preventive fungicide if necessary'
      ],
      'Squash___Powdery_mildew': [
        'Apply fungicide during the growing season',
        'Remove and destroy infected leaves',
        'Ensure proper spacing for air circulation',
        'Avoid overhead watering'
      ],
      'Strawberry___Leaf_scorch': [
        'Apply fungicide during the growing season',
        'Remove and destroy infected leaves',
        'Ensure proper spacing for air circulation',
        'Use resistant strawberry varieties'
      ],
      'Strawberry___healthy': [
        'Maintain good plant health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Ensure proper spacing for air circulation',
        'Apply preventive fungicide if necessary'
      ],
      'Tomato___Bacterial_spot': [
        'Apply copper-based bactericide during the growing season',
        'Remove and destroy infected plants',
        'Avoid overhead watering',
        'Use disease-free seeds and transplants'
      ],
      'Tomato___Early_blight': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant tomato varieties'
      ],
      'Tomato___Late_blight': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant tomato varieties'
      ],
      'Tomato___Leaf_Mold': [
        'Apply fungicide during the growing season',
        'Remove and destroy infected leaves',
        'Ensure proper spacing for air circulation',
        'Avoid overhead watering'
      ],
      'Tomato___Septoria_leaf_spot': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant tomato varieties'
      ],
      'Tomato___Spider_mites Two-spotted_spider_mite': [
        'Apply miticide during the growing season',
        'Remove and destroy heavily infested leaves',
        'Ensure proper irrigation to reduce stress',
        'Introduce natural predators like ladybugs'
      ],
      'Tomato___Target_Spot': [
        'Rotate crops to non-host crops',
        'Apply fungicide during the growing season',
        'Remove and destroy infected plant debris',
        'Use resistant tomato varieties'
      ],
      'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        'Control whitefly populations with insecticides',
        'Remove and destroy infected plants',
        'Use disease-free seeds and transplants',
        'Plant resistant tomato varieties'
      ],
      'Tomato___Tomato_mosaic_virus': [
        'Remove and destroy infected plants',
        'Control aphid populations with insecticides',
        'Use disease-free seeds and transplants',
        'Plant resistant tomato varieties'
      ],
      'Tomato___healthy': [
        'Maintain good plant health through proper watering and fertilization',
        'Regularly inspect for signs of disease',
        'Ensure proper spacing for air circulation',
        'Apply preventive fungicide if necessary'
      ]
    };
  
    // Return disease-specific recommendations if available, otherwise default
    return diseaseRecommendations[diseaseName] || defaultRecommendations;
  };