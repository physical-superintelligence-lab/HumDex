import React from 'react';
import AnimatedBackground from './components/AnimatedBackground';
import Section from './components/Section';
import VideoCarousel from './components/VideoCarousel';
import { CarouselItem } from './types';

const BIBTEX = `@article{humdex2026,
  title   = {HumDex: Humanoid Dexterous Manipulation Made Easy},
  author  = {Liang Heng and Yihe Tang and Jiajun Xu and Henghui Bao and Di Huang and Yue Wang},
  journal = {Conference Submission},
  year    = {2026}
}`;

const DEMO_VIDEOS: CarouselItem[] = [
  {
    id: 1,
    description: 'HumDex whole-body dexterous manipulation demo.',
    videoUrl: 'videos/demo.mp4',
    posterUrl: 'figs/demo.png',
  },
];

type ResultVideoItem = {
  id: number;
  title: string;
  videoUrl: string;
  featured?: boolean;
};

type GeneralizationVideoItem = {
  id: number;
  actor: 'Teleoperator' | 'Robot';
  generalization: 'Position' | 'Object' | 'Scene';
  title: string;
  videoUrl: string;
  rotateCCW?: boolean;
};

const INFERENCE_VIDEOS: ResultVideoItem[] = [
  { id: 1, title: 'Scan&Pack', videoUrl: 'videos/扫码-推理.hq2.mp4', featured: true },
  { id: 2, title: 'Hang Towel', videoUrl: 'videos/挂毛巾-推理.hq2.mp4' },
  { id: 3, title: 'Open Door', videoUrl: 'videos/开门-推理.hq2.mp4' },
  { id: 4, title: 'Place Basket', videoUrl: 'videos/提篮子-推理.hq2.mp4' },
  { id: 5, title: 'Pick Bread', videoUrl: 'videos/抓面包-推理-横.hq2.mp4' },
];

const TELEOP_VIDEOS: ResultVideoItem[] = [
  // Supports both local paths and full HTTPS URLs.
  { id: 1, title: 'Scan&Pack Teleoperation', videoUrl: 'videos/扫码-遥操-横.hq2.mp4' },
  { id: 2, title: 'Hang Towel Teleoperation', videoUrl: 'videos/挂毛巾-遥操-横.hq2.mp4' },
  { id: 3, title: 'Open Door Teleoperation', videoUrl: 'videos/开门-遥操.hq2.mp4' },
  { id: 4, title: 'Place Basket Teleoperation', videoUrl: 'videos/提篮子-遥操-横.hq2.mp4' },
  { id: 5, title: 'Pick Bread Teleoperation', videoUrl: 'videos/抓面包-遥操-横.hq2.mp4' },
];

const GENERALIZATION_VIDEOS: GeneralizationVideoItem[] = [
  {
    id: 1,
    actor: 'Teleoperator',
    generalization: 'Position',
    title: 'Teleoperator - Position #1',
    videoUrl: 'videos/人类数据采集-位置泛化1.mp4',
  },
  {
    id: 2,
    actor: 'Teleoperator',
    generalization: 'Object',
    title: 'Teleoperator - Object #1',
    videoUrl: 'videos/人类数据采集-物品泛化1.mp4',
    rotateCCW: true,
  },
  {
    id: 3,
    actor: 'Teleoperator',
    generalization: 'Scene',
    title: 'Teleoperator - Scene #1',
    videoUrl: 'videos/人类数据采集-背景泛化1.mp4',
    rotateCCW: true,
  },
  {
    id: 4,
    actor: 'Robot',
    generalization: 'Position',
    title: 'Robot - Position #1',
    videoUrl: 'videos/抓面包-推理-位置泛化1.mp4',
  },
  {
    id: 5,
    actor: 'Robot',
    generalization: 'Object',
    title: 'Robot - Object #1',
    videoUrl: 'videos/抓面包-推理-物品泛化1.mp4',
  },
  {
    id: 6,
    actor: 'Robot',
    generalization: 'Scene',
    title: 'Robot - Scene #1',
    videoUrl: 'videos/抓面包-推理-背景泛化1.mp4',
  },
  {
    id: 7,
    actor: 'Teleoperator',
    generalization: 'Position',
    title: 'Teleoperator - Position #2',
    videoUrl: 'videos/人类数据采集-位置泛化2.mp4',
  },
  {
    id: 8,
    actor: 'Teleoperator',
    generalization: 'Position',
    title: 'Teleoperator - Position #3',
    videoUrl: 'videos/人类数据采集-位置泛化3.mp4',
  },
  {
    id: 10,
    actor: 'Teleoperator',
    generalization: 'Object',
    title: 'Teleoperator - Object #2',
    videoUrl: 'videos/人类数据采集-物品泛化2.mp4',
  },
  {
    id: 11,
    actor: 'Teleoperator',
    generalization: 'Object',
    title: 'Teleoperator - Object #3',
    videoUrl: 'videos/人类数据采集-物品泛化3.mp4',
  },
  {
    id: 12,
    actor: 'Teleoperator',
    generalization: 'Scene',
    title: 'Teleoperator - Scene #2',
    videoUrl: 'videos/人类数据采集-背景泛化2.mp4',
  },
  {
    id: 13,
    actor: 'Teleoperator',
    generalization: 'Scene',
    title: 'Teleoperator - Scene #3',
    videoUrl: 'videos/人类数据采集-背景泛化3.mp4',
    rotateCCW: true,
  },
  {
    id: 14,
    actor: 'Robot',
    generalization: 'Position',
    title: 'Robot - Position #2',
    videoUrl: 'videos/抓面包-推理-位置泛化2.mp4',
  },
  {
    id: 15,
    actor: 'Robot',
    generalization: 'Position',
    title: 'Robot - Position #3',
    videoUrl: 'videos/抓面包-推理-位置泛化3.mp4',
  },
  {
    id: 17,
    actor: 'Robot',
    generalization: 'Object',
    title: 'Robot - Object #2',
    videoUrl: 'videos/抓面包-推理-物品泛化2.mp4',
  },
  {
    id: 18,
    actor: 'Robot',
    generalization: 'Object',
    title: 'Robot - Object #3',
    videoUrl: 'videos/抓面包-推理-物品泛化3.mp4',
  },
  {
    id: 19,
    actor: 'Robot',
    generalization: 'Scene',
    title: 'Robot - Scene #2',
    videoUrl: 'videos/抓面包-推理-背景泛化2.mp4',
  },
  {
    id: 20,
    actor: 'Robot',
    generalization: 'Scene',
    title: 'Robot - Scene #3',
    videoUrl: 'videos/抓面包-推理-背景泛化3.mp4',
  },
];

const App: React.FC = () => {
  const teleopScrollRef = React.useRef<HTMLDivElement>(null);
  const [selectedActors, setSelectedActors] = React.useState<Array<'Teleoperator' | 'Robot'>>([
    'Teleoperator',
  ]);
  const [selectedGeneralizations, setSelectedGeneralizations] = React.useState<
    Array<'Position' | 'Object' | 'Scene'>
  >(['Position']);
  const asset = (path: string) => `${import.meta.env.BASE_URL}${encodeURI(path)}`;
  const resolveVideoUrl = (url: string) =>
    /^https?:\/\//i.test(url) ? url : asset(url);
  const videoItems = DEMO_VIDEOS.map((item) => ({
    ...item,
    videoUrl: item.videoUrl ? resolveVideoUrl(item.videoUrl) : undefined,
    posterUrl: item.posterUrl ? resolveVideoUrl(item.posterUrl) : undefined,
  }));
  const inferenceVideos = INFERENCE_VIDEOS.map((item) => ({
    ...item,
    videoUrl: item.videoUrl ? resolveVideoUrl(item.videoUrl) : undefined,
  }));
  const teleopVideos = TELEOP_VIDEOS.map((item) => ({
    ...item,
    videoUrl: item.videoUrl ? resolveVideoUrl(item.videoUrl) : undefined,
  }));
  const generalizationVideos = GENERALIZATION_VIDEOS.map((item) => ({
    ...item,
    videoUrl: item.videoUrl ? resolveVideoUrl(item.videoUrl) : undefined,
  }));
  const toggleActor = (actor: 'Teleoperator' | 'Robot') => {
    setSelectedActors((prev) =>
      prev.includes(actor) ? prev.filter((x) => x !== actor) : [...prev, actor],
    );
  };
  const toggleGeneralization = (kind: 'Position' | 'Object' | 'Scene') => {
    setSelectedGeneralizations((prev) =>
      prev.includes(kind) ? prev.filter((x) => x !== kind) : [...prev, kind],
    );
  };
  const filteredGeneralizationVideos = generalizationVideos.filter(
    (item) =>
      selectedActors.includes(item.actor) &&
      selectedGeneralizations.includes(item.generalization),
  );
  const scrollTeleop = (direction: 'left' | 'right') => {
    const node = teleopScrollRef.current;
    if (!node) return;
    const offset = Math.round(node.clientWidth * 0.85);
    node.scrollBy({
      left: direction === 'left' ? -offset : offset,
      behavior: 'smooth',
    });
  };

  return (
    <div className="min-h-screen bg-brand-dark selection:bg-brand-cyan selection:text-brand-dark font-sans text-gray-100">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-brand-dark/80 backdrop-blur-md py-3 border-b border-white/5 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-full">
            <div className="flex-shrink-0 flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-purple to-brand-cyan flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-brand-purple/20">
                H
              </div>
              <a href="#hero" className="text-white text-xl font-bold tracking-tight hover:text-brand-cyan transition-colors">
                HumDex
              </a>
            </div>
            <div className="hidden md:flex space-x-1">
              {[
                ['Introduction', '#intro'],
                ['Method', '#method'],
                ['Results', '#results'],
                ['BibTeX', '#bibtex'],
              ].map(([name, href]) => (
                <a key={name} href={href} className="text-gray-300 hover:text-white hover:bg-white/10 px-4 py-2 rounded-full text-sm font-medium transition-all">
                  {name}
                </a>
              ))}
            </div>
          </div>
        </div>
      </nav>

      <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <AnimatedBackground />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-brand-dark/30 to-brand-dark pointer-events-none" />
        <div className="relative z-10 container mx-auto px-4 py-24 md:py-32 flex flex-col items-center text-center">
          <div className="mb-10 max-w-5xl">
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold tracking-tight leading-tight">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-white">
                HumDex
              </span>
              <span className="block mt-3 text-2xl md:text-4xl lg:text-5xl font-semibold text-gray-100">
                Humanoid Dexterous Manipulation Made Easy
              </span>
            </h1>
          </div>

          <p className="max-w-3xl text-base md:text-xl text-gray-300 leading-relaxed">
            Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang
          </p>
          <p className="mt-2 text-sm md:text-base text-gray-400">
            USC Physical Superintelligence (PSI) Lab
          </p>

          <div className="flex flex-wrap justify-center gap-4 mt-8">
            <a
              href={asset('571_HumDex_Humanoid_Dexterous_.pdf')}
              className="group inline-flex items-center gap-2 px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 hover:border-brand-cyan/50 text-white transition-all duration-300"
            >
              <i className="fas fa-file-pdf"></i>
              <span className="font-medium">Paper</span>
            </a>
            <a
              href="https://github.com/LiangHeng121/HumDex"
              target="_blank"
              rel="noopener noreferrer"
              className="group inline-flex items-center gap-2 px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 hover:border-brand-cyan/50 text-white transition-all duration-300"
            >
              <i className="fab fa-github"></i>
              <span className="font-medium">Code</span>
            </a>
            <a
              href="https://huggingface.co/heng222/humdex"
              target="_blank"
              rel="noopener noreferrer"
              className="group inline-flex items-center gap-2 px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 hover:border-brand-cyan/50 text-white transition-all duration-300"
            >
              <i className="fas fa-cube"></i>
              <span className="font-medium">Model</span>
            </a>
          </div>
        </div>
      </section>

      <main>
        <Section id="intro" title="Introduction" maxWidthClass="max-w-5xl">
          <div className="space-y-6">
            <div className="max-w-4xl mx-auto">
              <VideoCarousel items={videoItems} />
            </div>
            <p className="text-justify leading-relaxed text-gray-300">
              This paper investigates humanoid whole-body dexterous manipulation,
              where efficient collection of high-quality demonstrations remains a
              central bottleneck. We introduce HumDex, a portable teleoperation
              system that leverages IMU-based motion tracking for accurate
              full-body tracking and a learning-based hand retargeting method for
              smooth, natural dexterous control. Building on this system, we
              propose a two-stage imitation learning framework: pre-train on
              diverse human motion data, then fine-tune on robot data to bridge
              embodiment gaps. Experiments show strong improvements in collection
              efficiency, teleoperation success, downstream policy performance,
              and generalization to unseen positions, objects, and backgrounds.
            </p>
          </div>
        </Section>

        <Section id="method" title="Method Overview" maxWidthClass="max-w-5xl">
          <div className="space-y-6">
            <figure className="bg-black/30 rounded-xl overflow-hidden border border-white/10">
              <img
                src={asset('figs/method.jpg')}
                alt="Method Overview"
                className="w-full h-auto"
              />
            </figure>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <article className="bg-black/30 rounded-xl p-5 border border-white/10">
                <h3 className="text-xl font-bold mb-3">IMU-Based Whole-Body Teleoperation</h3>
                <p className="text-gray-300 text-sm leading-relaxed">
                  Portable motion tracking provides accurate full-body control
                  without heavy infrastructure or strict line-of-sight constraints.
                </p>
              </article>
              <article className="bg-black/30 rounded-xl p-5 border border-white/10">
                <h3 className="text-xl font-bold mb-3">Learning-Based Hand Retargeting</h3>
                <p className="text-gray-300 text-sm leading-relaxed">
                  A lightweight model maps glove fingertip observations to 20-DoF
                  hand joints, producing smooth dexterous motions without manual tuning.
                </p>
              </article>
              <article className="bg-black/30 rounded-xl p-5 border border-white/10">
                <h3 className="text-xl font-bold mb-3">Two-Stage Imitation Learning</h3>
                <p className="text-gray-300 text-sm leading-relaxed">
                  The policy is first trained on diverse human data, then fine-tuned
                  on robot demonstrations for embodiment-specific precision.
                </p>
              </article>
            </div>
          </div>
        </Section>

        <Section id="results" title="Results" maxWidthClass="max-w-5xl">
          <div className="flex flex-col gap-6">
            <div className="order-2">
              <h3 className="text-2xl font-bold mb-4">Autonomous Inference</h3>
              <p className="text-sm text-gray-300 mb-3">
                We evaluate autonomous policy performance on five representative tasks:
                Open Door, Scan&Pack, Place Basket, Hang Towel, and Pick Bread.
              </p>
              <div className="space-y-4">
                {(() => {
                  const openDoor = inferenceVideos.find((v) => v.title === 'Open Door');
                  const placeBasket = inferenceVideos.find((v) => v.title === 'Place Basket');
                  const scanPack = inferenceVideos.find((v) => v.title === 'Scan&Pack');
                  const hangTowel = inferenceVideos.find((v) => v.title === 'Hang Towel');
                  const pickBread = inferenceVideos.find((v) => v.title === 'Pick Bread');
                  return (
                    <>
                      <div className="grid grid-cols-1 md:grid-cols-[0.72fr_1.14fr_1.14fr] gap-4 md:grid-rows-[1.7fr_1fr] md:h-[560px] md:w-[90%] md:mx-auto">
                        <div className="grid grid-cols-1 grid-rows-2 gap-4 md:col-start-1 md:row-span-2 md:row-start-1 h-full min-h-0">
                          {openDoor && (
                            <figure
                              key={openDoor.id}
                              className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10 h-full min-h-0"
                            >
                              <video
                                className="w-full h-full object-cover"
                                muted
                                loop
                                playsInline
                                controls
                                preload="auto"
                              >
                                <source src={openDoor.videoUrl} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                              <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-black/60 px-3 py-1 rounded-md border border-white/20">
                                {openDoor.title}
                              </figcaption>
                            </figure>
                          )}
                          {placeBasket && (
                            <figure
                              key={placeBasket.id}
                              className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10 h-full min-h-0"
                            >
                              <video
                                className="w-full h-full object-cover"
                                muted
                                loop
                                playsInline
                                controls
                                preload="auto"
                              >
                                <source src={placeBasket.videoUrl} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                              <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-black/60 px-3 py-1 rounded-md border border-white/20">
                                {placeBasket.title}
                              </figcaption>
                            </figure>
                          )}
                        </div>
                        {scanPack && (
                          <figure
                            key={scanPack.id}
                            className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10 md:col-start-2 md:col-span-2 md:row-start-1 h-full min-h-0"
                          >
                            <video
                              className="w-full h-full object-cover"
                              muted
                              loop
                              playsInline
                              controls
                              preload="auto"
                            >
                              <source src={scanPack.videoUrl} type="video/mp4" />
                              Your browser does not support the video tag.
                            </video>
                            <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-black/60 px-3 py-1 rounded-md border border-white/20">
                              {scanPack.title}
                            </figcaption>
                          </figure>
                        )}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:col-start-2 md:col-span-2 md:row-start-2 h-full min-h-0">
                          {[hangTowel, pickBread].filter(Boolean).map((item) => (
                            <figure
                              key={item!.id}
                              className="relative w-full h-full min-h-0 bg-black/30 rounded-xl overflow-hidden border border-white/10"
                            >
                              <video
                                className="w-full h-full object-cover"
                                muted
                                loop
                                playsInline
                                controls
                                preload="auto"
                              >
                                <source src={item!.videoUrl} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                              <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-black/60 px-3 py-1 rounded-md border border-white/20">
                                {item!.title}
                              </figcaption>
                            </figure>
                          ))}
                        </div>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>

            <div className="order-1">
              <h3 className="text-2xl font-bold mb-4">Teleoperation</h3>
              <p className="text-sm text-gray-300 mb-3">
                We present representative teleoperation demonstrations for Scan&Pack, Hang Towel,
                Open Door, Place Basket, and Pick Bread.
              </p>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  aria-label="Scroll left"
                  onClick={() => scrollTeleop('left')}
                  className="shrink-0 h-12 w-12 rounded-full border border-white/20 bg-black/40 hover:bg-black/60 transition-colors flex items-center justify-center text-xl"
                >
                  ‹
                </button>
                <div
                  ref={teleopScrollRef}
                  className="flex gap-4 overflow-x-auto snap-x snap-mandatory flex-1 min-w-0 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
                >
                  {teleopVideos.map((item) => (
                    <figure
                      key={item.id}
                      className="relative shrink-0 w-[460px] sm:w-[540px] snap-start bg-black/30 rounded-xl overflow-hidden border border-white/10"
                    >
                      <video
                        className="w-full aspect-video object-cover"
                        muted
                        loop
                        playsInline
                        controls
                        preload="auto"
                      >
                        <source src={item.videoUrl} type="video/mp4" />
                        Your browser does not support the video tag.
                      </video>
                      <figcaption className="absolute top-2 left-2 text-sm font-semibold bg-black/60 px-3 py-1 rounded-md border border-white/20">
                        {item.title}
                      </figcaption>
                    </figure>
                  ))}
                </div>
                <button
                  type="button"
                  aria-label="Scroll right"
                  onClick={() => scrollTeleop('right')}
                  className="shrink-0 h-12 w-12 rounded-full border border-white/20 bg-black/40 hover:bg-black/60 transition-colors flex items-center justify-center text-xl"
                >
                  ›
                </button>
              </div>
            </div>

            <div className="order-3">
              <h3 className="text-2xl font-bold mb-4">Bread Generalization (Egocentric View)</h3>
              <p className="text-sm text-gray-300 mb-3">
                We evaluate policy generalization across positions, objects, and scenes from an
                egocentric view for both Teleoperator and Robot settings.
              </p>
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  {(['Teleoperator', 'Robot'] as const).map((actor) => {
                    const active = selectedActors.includes(actor);
                    return (
                      <button
                        key={actor}
                        type="button"
                        onClick={() => toggleActor(actor)}
                        className={`px-4 py-2 rounded-full border text-sm transition-colors ${
                          active
                            ? 'bg-cyan-400/20 border-cyan-300/50 text-cyan-100'
                            : 'bg-black/30 border-white/20 text-gray-300 hover:bg-black/50'
                        }`}
                      >
                        {actor}
                      </button>
                    );
                  })}
                </div>
                <div className="flex flex-wrap gap-2">
                  {(['Position', 'Object', 'Scene'] as const).map((kind) => {
                    const active = selectedGeneralizations.includes(kind);
                    return (
                      <button
                        key={kind}
                        type="button"
                        onClick={() => toggleGeneralization(kind)}
                        className={`px-4 py-2 rounded-full border text-sm transition-colors ${
                          active
                            ? 'bg-purple-400/20 border-purple-300/50 text-purple-100'
                            : 'bg-black/30 border-white/20 text-gray-300 hover:bg-black/50'
                        }`}
                      >
                        {kind} Generalization
                      </button>
                    );
                  })}
                </div>

                {filteredGeneralizationVideos.length === 0 ? (
                  <p className="text-sm text-gray-400">
                    No video selected. Please choose at least one actor and one
                    generalization type.
                  </p>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {filteredGeneralizationVideos.map((item) => (
                      <figure
                        key={item.id}
                        className="relative bg-black/30 rounded-xl overflow-hidden border border-white/10"
                      >
                        <div className="w-full aspect-video overflow-hidden bg-black flex items-center justify-center">
                          <video
                            className="w-full h-full object-cover"
                            muted
                            loop
                            playsInline
                            controls
                            preload="auto"
                          >
                            <source src={item.videoUrl} type="video/mp4" />
                            Your browser does not support the video tag.
                          </video>
                        </div>
                        <figcaption className="absolute top-2 left-2 text-xs font-semibold bg-black/60 px-2 py-1 rounded border border-white/20">
                          {item.actor} / {item.generalization}
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                )}
              </div>
            </div>

          </div>
        </Section>

        <Section id="bibtex" title="BibTeX">
          <pre className="bg-[#0d1117] p-6 rounded-xl overflow-x-auto text-sm text-gray-300 font-mono border border-white/10 shadow-inner">
            {BIBTEX}
          </pre>
        </Section>
      </main>

      <footer className="bg-black/80 backdrop-blur-md border-t border-white/5 py-12 mt-20">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-500 text-sm">
            © {new Date().getFullYear()} HumDex Project. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;