import React, { useState } from 'react';
import { CarouselItem } from '../types';

interface VideoCarouselProps {
  items: CarouselItem[];
}

const VideoCarousel: React.FC<VideoCarouselProps> = ({ items }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [activatedYoutube, setActivatedYoutube] = useState<Record<string, boolean>>({});
  const [isLocalPlaying, setIsLocalPlaying] = useState<Record<string, boolean>>({});

  const nextSlide = () => {
    setActivatedYoutube({});
    setCurrentIndex((prev) => (prev + 1) % items.length);
  };

  const prevSlide = () => {
    setActivatedYoutube({});
    setCurrentIndex((prev) => (prev - 1 + items.length) % items.length);
  };

  const goToSlide = (index: number) => {
    setActivatedYoutube({});
    setCurrentIndex(index);
  };

  const toggleLocalVideoPlay: React.MouseEventHandler<HTMLVideoElement> = (event) => {
    const video = event.currentTarget;
    if (video.paused) {
      video.play().catch(() => {
        // Ignore interrupted play attempts.
      });
      return;
    }
    video.pause();
  };

  return (
    <div className="relative w-full">
      {/* Main Content Area */}
      <div className="relative overflow-hidden rounded-xl aspect-video bg-black shadow-lg border border-white/10 group">
        
        {items.map((item, index) => (
          <div
            key={item.id}
            className={`absolute inset-0 transition-opacity duration-500 ease-in-out flex flex-col items-center justify-center ${
              index === currentIndex ? 'opacity-100 z-10' : 'opacity-0 z-0'
            }`}
          >
            {/* Render YouTube or Local Video */}
            {item.youtubeId ? (
              activatedYoutube[String(item.id)] ? (
                <iframe
                  className="w-full h-full"
                  src={`https://www.youtube-nocookie.com/embed/${item.youtubeId}?autoplay=1&rel=0&modestbranding=1&iv_load_policy=3&playsinline=1&controls=0&fs=0&disablekb=1`}
                  title={item.title || 'Video'}
                  allow="autoplay; encrypted-media; picture-in-picture"
                  allowFullScreen={false}
                />
              ) : (
                <button
                  type="button"
                  onClick={() =>
                    setActivatedYoutube((prev) => ({
                      ...prev,
                      [String(item.id)]: true,
                    }))
                  }
                  className="relative w-full h-full focus:outline-none cursor-pointer"
                  aria-label={`Play ${item.title || 'video'}`}
                >
                  <img
                    src={item.posterUrl || `https://i.ytimg.com/vi/${item.youtubeId}/maxresdefault.jpg`}
                    alt={item.title || 'Video thumbnail'}
                    className="w-full h-full object-cover"
                    onError={(event) => {
                      (event.currentTarget as HTMLImageElement).src = `https://i.ytimg.com/vi/${item.youtubeId}/sddefault.jpg`;
                    }}
                  />
                </button>
              )
            ) : item.videoUrl ? (
              <div className="relative w-full h-full">
                <video
                  className="w-full h-full object-cover cursor-pointer"
                  src={item.videoUrl}
                  playsInline
                  preload="auto"
                  onClick={toggleLocalVideoPlay}
                  onPlay={() =>
                    setIsLocalPlaying((prev) => ({
                      ...prev,
                      [String(item.id)]: true,
                    }))
                  }
                  onPause={() =>
                    setIsLocalPlaying((prev) => ({
                      ...prev,
                      [String(item.id)]: false,
                    }))
                  }
                >
                  Your browser does not support the video tag.
                </video>
                {/* No overlay on hover by request */}
              </div>
            ) : (
                <div className="flex items-center justify-center w-full h-full text-gray-500">
                    No Video Source
                </div>
            )}

            {/* Caption removed by request */}
          </div>
        ))}

      </div>

      {/* Dots/Indicators */}
      <div className="flex justify-center mt-4 gap-2">
        {items.map((_, index) => (
          <button
            key={index}
            onClick={() => goToSlide(index)}
            className={`transition-all duration-300 rounded-full ${
              index === currentIndex
                ? 'w-8 h-2 bg-brand-cyan'
                : 'w-2 h-2 bg-gray-600 hover:bg-gray-400'
            }`}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
};

export default VideoCarousel;