"use client";

import React, { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  securityLevel: "loose",
});

export default function Mermaid({ chart }: { chart: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svgContent, setSvgContent] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        observer.disconnect();
        
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        mermaid
          .render(id, chart)
          .then(({ svg, bindFunctions }) => {
            setSvgContent(svg);
            if (bindFunctions && containerRef.current) {
              // Note: bindFunctions might need the DOM element to exist,
              // but for standard charts it's often fine without.
            }
          })
          .catch((err) => {
            setError(err.message || String(err));
          });
      }
    });

    observer.observe(containerRef.current);

    return () => observer.disconnect();
  }, [chart]);

  if (error) {
    return (
      <div className="p-4 bg-red-950 text-red-200 rounded-md text-sm overflow-x-auto">
        <pre>{error}</pre>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="mermaid-flow-wrap p-4 overflow-x-auto flex justify-center w-full min-h-[150px]"
      dangerouslySetInnerHTML={{ __html: svgContent }}
    />
  );
}
