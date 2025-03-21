import React from 'react';

const ServiceCard = ({ service }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-center text-center transition-transform hover:transform hover:scale-105">
      <div className="text-4xl mb-4">{service.icon}</div>
      <h3 className="text-xl font-semibold text-teal-800 mb-2">{service.title}</h3>
      <p className="text-gray-600">{service.description}</p>
    </div>
  );
};

export default ServiceCard;