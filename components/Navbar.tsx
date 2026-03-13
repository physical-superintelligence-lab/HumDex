import React, { useState, useEffect } from 'react';

const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
      { name: 'Intro', href: '#intro' },
      { name: 'Method', href: '#method' },
      { name: 'Results', href: '#results' },
  ];

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled ? 'bg-brand-dark/80 backdrop-blur-md py-3 border-b border-white/5 shadow-lg' : 'bg-transparent py-5'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-full">
          {/* Logo */}
          <div className="flex-shrink-0 flex items-center gap-2">
             <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-purple to-brand-cyan flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-brand-purple/20">
                V
             </div>
             <a href="#hero" className="text-white text-xl font-bold tracking-tight hover:text-brand-cyan transition-colors">
               HumDex
             </a>
          </div>

          {/* Desktop Nav */}
          <div className="hidden md:flex space-x-1">
            {navLinks.map((link) => (
              <a 
                key={link.name}
                href={link.href} 
                className="text-gray-300 hover:text-white hover:bg-white/10 px-4 py-2 rounded-full text-sm font-medium transition-all"
              >
                {link.name}
              </a>
            ))}
            
            <div className="w-px h-6 bg-white/10 mx-2 self-center"></div>

            <a href="https://general-flow.github.io/" target="_blank" rel="noopener noreferrer" className="text-brand-cyan hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
              <i className="fas fa-external-link-alt mr-2 text-xs"></i>Related Research
            </a>
          </div>
          
          {/* Mobile menu button could go here */}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;