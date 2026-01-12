import React, { useEffect, useRef, useMemo } from 'react';

const Demos = () => {
  const videoContainersRef = useRef([]);
  const videos = useMemo(() => [
    { url: 'pythia_overview.mp4', duration: 'xxx min', title: '1. Workflow Overview', description: 'How the five-agent architecture collaborates to optimize prompts for cognitive concern detection without human-in-the-loop tweaks.' }
  ], []);

  useEffect(() => {
    const loadVideo = (container, videoInfo) => {
      if (!container) return;

      const videoUrl = videoInfo.url;
      const videoPath = `/assets/videos/${videoUrl}`;
      const video = document.createElement('video');
      video.controls = true;
      video.autoplay = true;
      video.muted = true;
      video.loop = true;
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.objectFit = 'cover';
      video.src = videoPath;

      video.onloadeddata = () => {
        container.innerHTML = '';
        container.appendChild(video);
        video.play().catch(error => console.error("Autoplay was prevented: ", error));
      };

      video.onerror = () => {
        container.innerHTML = `
          <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: #000; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; flex-direction: column; gap: 16px;">
              <div style="font-size: 48px;">🎥</div>
              <div>Video not found: ${videoUrl}</div>
              <div style="font-size: 14px; opacity: 0.7;">Place videos in <code>public/assets/videos</code></div>
          </div>
        `;
      };
    };

    videoContainersRef.current.forEach((container, index) => {
      loadVideo(container, videos[index]);
    });
  }, [videos]);

 return (
    <section className="demo-section" id="demos">
      <div className="container">
        <div className="section-header fade-in">
          <h2>What is Pythia?</h2>
          <p>Pythia is an iterative tool for allowing your LLM to improve upon it's own prompts. Instead of manually changing and perfecting your own prompt, Pythia can do it for you, by testing on a dataset you provide and determining where it needs to improve.</p>
          <br />
          <p> Pythia works by first calling the Specialist agent to test your prompt, and evaluates the performance metrics compared to the provided ground truth answer. It will then route to the corresponding improvement agent (sensitivity, or specificity) based on your results and priority. For each false positive, or false negative individual, it will process the mistaken notes to figure out where the LLM went wrong, and provide evidence of the true classification of the note for the summarizer to build it's new prompt on. From there, the summarizer will be passed the evidence, as well as the original prompt and SOP to build the new prompt. Once a new prompt is created, it will repeat the process until it reaches the maximum iterations or meets the performance thresholds. From there, it will validate the new prompt on the development dataset.</p>
        </div>
      </div>
    </section>
  );
};

export default Demos;

