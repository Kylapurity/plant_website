import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Testimonial from './Testimonial';
import ServiceCard from './ServiceCard';
import BlogCard from './BlogCard';
import Footer from './Footer';
import Plant from '../assets/crop-protection.jpg';
import Tomate from '../assets/Tomate.jpg';
// Add these new imports for blog images
import PotatoBlight from '../assets/potato-diseases.jpg';
import ComputerVision from '../assets/maize.jpg';
import SustainablePractices from '../assets/Robust.jpg';
import Grow from '../assets/Grow.jpg';
import Fresh from '../assets/fresht.jpg';
import Miller from '../assets/miller.jpg';
import Apples from '../assets/apple.jpg';
import Sale from '../assets/Agri.jpg';
import Refresh from '../assets/detection.jpg';
import Next from '../assets/womanb.jpg';
import White1 from '../assets/white.png'
// Import Slider from react-slick
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

const HomePage = () => {
  // Add the missing state for tracking scroll position
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 0;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [scrolled]);
  
  const services = [
    {
      id: 1,
      title: 'Plant Analysis',
      description: 'Advanced image recognition technology to identify plant diseases',
      icon: '📊'
    },
    {
      id: 2,
      title: 'Treatment Guidance',
      description: 'Personalized recommendations based on detected conditions',
      icon: '🔍'
    },
    {
      id: 3,
      title: 'Disease Tracking',
      description: 'Monitor plant health and disease progression over time',
      icon: '📈'
    },
    {
      id: 4,
      title: 'Expert Consultations',
      description: 'Connect with agricultural specialists for complex cases',
      icon: '👨‍🌾'
    }
  ];

  const testimonials = [
    {
      id: 1,
      name: 'Sarah Johnson',
      role: 'Organic Farmer',
      image: '/api/placeholder/64/64',
      text: 'This plant disease detection tool has revolutionized how I manage pest control on my farm. The early detection has saved countless crops!'
    },
    {
      id: 2,
      name: 'Michael Chen',
      role: 'Greenhouse Manager',
      image: '/api/placeholder/64/64',
      text: 'The accuracy of disease identification is impressive. Now we can take preventative measures before diseases spread throughout our greenhouse.'
    },
    {
      id: 3,
      name: 'Ali Mark',
      role: 'Domestic Farmer',
      image: '/api/placeholder/64/64',
      text: 'The accuracy of disease identification is impressive. Now we can take preventative measures before diseases spread throughout our greenhouse.'
    }
  ];

  const blogPosts = [
    {
      id: 1,
      title: 'Early Detection of Potate Blight',
      image: PotatoBlight,
      excerpt: 'Learn how to identify the early signs of tomato blight and take action before it spreads to your entire garden.',
      url: '#'
    },
    {
      id: 2,
      title: 'The Power of Computer Vision in Agriculture',
      image: ComputerVision,
      excerpt: 'Discover how machine learning algorithms are changing the way farmers diagnose and treat plant diseases.',
      url: '#'
    },
    {
      id: 3,
      title: 'Sustainable Disease Management Practices',
      image: SustainablePractices,
      excerpt: 'Explore eco-friendly approaches to managing plant diseases without harmful chemicals.',
      url: '#'
    }
  ];

  // New hero slides array with three slides
  const heroSlides = [
    {
      id: 1,
      image: Next,
      title: "Providing Nature's Finest Harvest",
      subtitle: "Established Since 1992",
      description: "Semper sit imperdiet bibendum felis fames ante sed suscipit tristique ultrices aptent mi sagittis.",
      buttonText: "Discover More"
    },
    {
      id: 2,
      image: Refresh,
      title: "Sustainable Farming Solutions",
      subtitle: "Organic Since 1992",
      description: "Growing the healthiest produce with respect for nature and sustainable farming practices.",
      buttonText: "Learn More"
    },
    {
      id: 3,
      image: White1,
      title: "AI-Powered Agriculture",
      subtitle: "The Future of Farming",
      description: "Leveraging cutting-edge technology to maximize yields and minimize environmental impact.",
      buttonText: "Explore Now"
    }
  ];

  // Settings for the react-slick slider for testimonials
  const sliderSettings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 3,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 5000,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 1,
          infinite: true,
          dots: true
        }
      },
      {
        breakpoint: 768,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1,
          initialSlide: 1
        }
      }
    ]
  };

  // Settings for the hero slider
  const heroSliderSettings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 6000,
    fade: true,
    cssEase: 'linear'
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation - Reversed the logic: now white when not scrolled, transparent when scrolled */}
      <nav className={`fixed w-full py-4 px-10 transition-all duration-500 z-50 ${scrolled ? 'bg-green-800' : 'bg-white shadow-lg'}`}>
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className={`text-xl font-bold ${scrolled ? 'text-white' : 'text-teal-700'}`}>Smart Farm</h1>
          <div className="flex space-x-10">
            <a href="#" className={`hover:text-teal-700 ${scrolled ? 'text-white' : 'text-gray-700'}`}>Home</a>
            <a href="#services" className={`hover:text-teal-700 ${scrolled ? 'text-white' : 'text-gray-700'}`}>Services</a>
            <a href="#about" className={`hover:text-teal-700 ${scrolled ? 'text-white' : 'text-gray-700'}`}>About Us</a>
            <a href="#blog" className={`hover:text-teal-700 ${scrolled ? 'text-white' : 'text-gray-700'}`}>Blog</a>
          </div>
          <div className="flex space-x-4">
            <Link
              to="/login"
              className={`px-4 py-2 border text-base font-medium rounded-md ${scrolled ? 'text-white bg-green-800 border-white' : 'text-green-700 bg-white'} hover:bg-gray-50 hover:text-green-700 shadow-md`}
            >
              Log in
            </Link>
            <Link
              to="/signup"
              className="px-4 py-2 border text-base font-medium rounded-md text-white bg-green-800 hover:bg-green-900 shadow-md"
            >
              Sign up
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section with Slider - Fixed for full visibility */}
      <section className="relative w-full h-[600px] overflow-hidden pt-16">
        <Slider {...heroSliderSettings} className="h-full">
          {heroSlides.map((slide) => (
            <div key={slide.id} className="relative h-[600px]">
              <img 
                src={slide.image}
                alt={`Hero Background ${slide.id}`} 
                className="absolute inset-0 w-full h-full object-cover"
              />
              {/* Overlay to improve text readability */}
              <div className="absolute inset-0 opacity-30" /> 
              <div className="absolute inset-0 flex items-start pt-24">
                <div className="container mx-auto px-6 md:px-12 text-left">
                  <div className="max-w-xl">
                    <div className="flex items-center mb-4">
                      <span className="text-green-400 mr-2">🌱</span>
                      <p className="text-green-400 font-medium">{slide.subtitle}</p>
                    </div>
                    <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4 leading-tight text-white">
                      {slide.title}
                    </h1>
                    <p className="text-base md:text-lg mb-6 text-gray-200">
                      {slide.description}
                    </p>
                    <a href="#learn-more" className="px-6 py-3 rounded-md bg-green-800 text-white font-medium hover:bg-green-700 inline-block">
                      {slide.buttonText}
                    </a>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </Slider>
      </section>
      
      {/* About Section */}
      <section id="about" className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center">
            <div className="md:w-1/2 mb-10 md:mb-0 md:pr-10">
              <div className="inline-block bg-teal-100 px-2 py-1 text-teal-800 text-sm font-medium rounded-md mb-4">Since 2024</div>
              <h2 className="text-3xl font-bold text-teal-800 mb-6">
                Our Passion for Protecting Plants and Ensuring Agricultural Success
              </h2>
              <p className="text-gray-700 mb-6">
                PlantHealth AI combines cutting-edge artificial intelligence with agricultural expertise to offer farmers, gardeners, and plant enthusiasts a reliable tool for disease detection and management.
              </p>
              <p className="text-gray-700 mb-6">
                Our mission is to reduce crop losses, minimize pesticide use, and promote sustainable farming practices through accurate and timely plant disease diagnosis.
              </p>
              <Link to="/signup" className="inline-block px-6 py-3 bg-teal-700 text-white rounded-md hover:bg-teal-800">
                Get Started
              </Link>
            </div>
            <div className="md:w-1/2">
              <img src={Tomate} alt="Healthy crops" className="rounded-lg shadow-lg" />
            </div>
          </div>
        </div>
      </section>

      {/* Services Section */}
      <section id="services" className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-teal-800 mb-4">Our Services</h2>
            <p className="text-gray-700 max-w-3xl mx-auto">
              We offer a comprehensive suite of plant disease detection and management tools powered by advanced machine learning algorithms.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {services.map(service => (
              <ServiceCard key={service.id} service={service} />
            ))}
          </div>
        </div>
      </section>

      <section className="flex flex-col md:flex-row items-center justify-between p-10 bg-white">
      {/* Left Section - Image */}
      <div className="w-full md:w-1/3">
        <img
          src={Sale}
          alt="Organic Farm"
          className="rounded-lg shadow-lg"
        />
      </div>
      
      {/* Middle Section - Text */}
      <div className="w-full md:w-1/3 text-center md:text-left px-8">
        <h2 className="text-3xl font-bold mb-4">Best Organic Agriculture Firms</h2>
        <p className="text-gray-600 mb-4">
          Give lady of they such they sure it. Me contained explained my education.
          Vulgar as hearts by garret. Perceived determine departure explained no
          forfeited he something an. Contrasted dissimilar get joy you instrument out
          reasonably. Again keeps at no meant stuff.
        </p>
        <div className="mb-4">
          <p className="flex items-center gap-2">
            ✅ <span className="font-semibold">Best Quality Standards</span>
          </p>
          <p className="flex items-center gap-2">
            ✅ <span className="font-semibold">Natural Healthy Products</span>
          </p>
        </div>
        <button className="bg-yellow-500 text-white px-6 py-2 rounded-lg shadow-md hover:bg-yellow-600">
          Discover More
        </button>
      </div>
      
      {/* Right Section - List with Images */}
      <div className="w-full md:w-1/4 ">
        <ul className="space-y-4">
          <li className="flex items-center gap-2">
            <img src={Grow} alt="Fresh Potatoes" className="w-12 h-12 rounded-full" />
            <span className="font-semibold">Fresh Potatoes</span>
          </li>
          <li className="flex items-center gap-2">
            <img src={Tomate} alt="Healthy Tomatoes" className="w-12 h-12 rounded-full" />
            <span className="font-semibold">Healthy Tomatoes</span>
          </li>
          <li className="flex items-center gap-2">
            <img src={Miller} alt="Organic Maize" className="w-12 h-12 rounded-full" />
            <span className="font-semibold">Organic Maize</span>
          </li>
          <li className="flex items-center gap-2">
            <img src={Apples} alt="Juicy Apples" className="w-12 h-12 rounded-full" />
            <span className="font-semibold">Juicy Apples</span>
          </li>
        </ul>
      </div>
    </section>

      {/* Testimonials */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-teal-800 mb-4">Testimonials</h2>
            <p className="text-gray-700 max-w-3xl mx-auto">
              Hear what our users have to say about how our plant disease detection platform has helped their agricultural endeavors.
            </p>
          </div>
          <div className="relative">
            <Slider {...sliderSettings}>
              {testimonials.map(testimonial => (
                <div key={testimonial.id} className="px-2">
                  <Testimonial testimonial={testimonial} />
                </div>
              ))}
            </Slider>
          </div>
        </div>
      </section>

      {/* Blog Section */}
      <section id="blog" className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-teal-800 mb-4">Our Blog</h2>
            <p className="text-gray-700 max-w-3xl mx-auto">
              Explore our latest articles on plant diseases, agriculture technology, and sustainable farming practices.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {blogPosts.map(post => (
              <BlogCard key={post.id} post={post} />
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default HomePage;