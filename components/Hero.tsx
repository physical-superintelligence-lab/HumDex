import React from 'react';
import { Author, Affiliation, LinkButton } from '../types';
import AnimatedBackground from './AnimatedBackground';

// --- Data ---
const AUTHORS: Author[] = [
  { name: "Yueru Jia", url: "https://jiayueru.github.io/", affiliations: [1, 2], isEqualContribution: true },
  { name: "Jiaming Liu", url: "https://liujiaming1996.github.io/", affiliations: [1, 2], isEqualContribution: true },
  { name: "Shengbang Liu", affiliations: [3], isEqualContribution: true },
  { name: "Rui Zhou", url: "https://zhourui9813.github.io/", affiliations: [4] },
  { name: "Wanhe Yu", affiliations: [1] },
  { name: "Yuyang Yan", affiliations: [1] },
  { name: "Xiaowei Chi", affiliations: [5] },
  { name: "Yandong Guo", affiliations: [2] },
  { name: "Boxin Shi", url: "https://camera.pku.edu.cn/", affiliations: [1] },
  { name: "Shanghang Zhang", url: "https://www.shanghangzhang.com/", affiliations: [1], isCorresponding: true },
];

const AFFILIATIONS: Affiliation[] = [
  { id: 1, name: "State Key Laboratory of Multimedia Information Processing, School of Computer Science" },
  { id: 2, name: "AI2 Robotics" },
  { id: 3, name: "Sun Yat-sen University" },
  { id: 4, name: "Wuhan University" },
  { id: 5, name: "Hong Kong University of Science and Technology" },
];

const LINKS: LinkButton[] = [
  { label: "Paper (PDF)", iconClass: "fas fa-file-pdf", url: "./resources/" },
  { label: "arXiv", iconClass: "ai ai-arxiv", url: "https://arxiv.org/" },
  { label: "Code", iconClass: "fab fa-github", url: "https://github.com/" },
  { label: "VLA Code", iconClass: "fas fa-code", url: "https://github.com/" },
  { label: "Datasets", iconClass: "fas fa-database", url: "https://huggingface.co/", isImage: true }, // Using generic icon as fallback for HF image
];

const Hero: React.FC = () => {
  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Layer */}
      <AnimatedBackground />
      
      {/* Gradient Overlay for Depth */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-brand-dark/30 to-brand-dark pointer-events-none" />

      {/* Main Content */}
      <div className="relative z-10 container mx-auto px-4 py-24 md:py-32 flex flex-col items-center text-center">
        
        {/* Title Section */}
        <div className="mb-12 max-w-5xl animate-[fadeIn_1s_ease-out]">
            <h1 className="flex flex-col items-center gap-4">
                <span className="relative inline-block">
                    {/* Glowing effect behind the main brand */}
                    <span className="absolute -inset-1 rounded-lg bg-gradient-to-r from-brand-purple to-brand-cyan opacity-50 blur-xl animate-pulse"></span>
                    <span className="relative text-6xl md:text-8xl lg:text-9xl font-extrabold tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-white drop-shadow-sm">
                        Video2Act
                    </span>
                </span>
                
                <span className="text-2xl md:text-3xl lg:text-4xl font-semibold text-gray-100 mt-4 leading-tight">
                    A Dual-System Video Diffusion Policy <br className="hidden md:block"/>
                    with <span className="text-brand-cyan">Robotic Spatio-Motional Modeling</span>
                </span>
            </h1>
        </div>

        {/* Authors Section */}
        <div className="mb-8 max-w-4xl mx-auto backdrop-blur-sm bg-white/5 rounded-2xl p-6 border border-white/10 shadow-2xl">
          <div className="flex flex-wrap justify-center gap-x-4 gap-y-2 text-lg md:text-xl leading-relaxed text-gray-200">
            {AUTHORS.map((author, index) => (
              <span key={index} className="inline-flex items-center">
                {author.url ? (
                  <a href={author.url} target="_blank" rel="noopener noreferrer" className="font-semibold text-brand-cyan hover:text-white transition-colors duration-200 hover:underline decoration-brand-purple decoration-2 underline-offset-4">
                    {author.name}
                  </a>
                ) : (
                  <span className="font-medium">{author.name}</span>
                )}
                <sup className="ml-0.5 text-sm text-brand-purple font-bold">
                  {author.affiliations.join(',')}
                  {author.isEqualContribution && '*'}
                  {author.isCorresponding && '†'}
                </sup>
              </span>
            ))}
          </div>

          {/* Affiliations */}
          <div className="mt-6 text-sm md:text-base text-gray-400 space-y-1">
             <div className="flex flex-wrap justify-center gap-x-6 gap-y-1">
                {AFFILIATIONS.map(aff => (
                    <span key={aff.id}>
                        <sup className="text-brand-purple mr-1">{aff.id}</sup>
                        {aff.name}
                    </span>
                ))}
             </div>
             <div className="pt-3 text-xs md:text-sm text-gray-500 font-mono">
                * Equal contribution &nbsp;&nbsp; † Corresponding author
             </div>
          </div>
        </div>

        {/* Buttons / Links */}
        <div className="flex flex-wrap justify-center gap-4 mt-4">
          {LINKS.map((link, idx) => (
            <a
              key={idx}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group relative inline-flex items-center gap-2 px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 hover:border-brand-cyan/50 text-white transition-all duration-300 hover:-translate-y-1 shadow-lg hover:shadow-brand-cyan/20 backdrop-blur-md"
            >
              <span className="text-xl group-hover:text-brand-cyan transition-colors">
                 {link.isImage ? <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" className="w-5 h-5" /> : <i className={link.iconClass}></i>}
              </span>
              <span className="font-medium tracking-wide">{link.label}</span>
            </a>
          ))}
        </div>

      </div>
      
      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce text-white/50">
        <i className="fas fa-chevron-down text-2xl"></i>
      </div>
    </section>
  );
};

export default Hero;