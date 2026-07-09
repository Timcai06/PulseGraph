import gsap from "gsap";

export const motionDurations = {
  quick: 0.18,
  enter: 0.36,
  panel: 0.5,
  drawer: 0.55,
  signal: 0.42,
  count: 0.7
} as const;

export const motionEase = {
  standard: "power2.out",
  panel: "power3.inOut",
  signalIn: "power1.in",
  signalOut: "power1.out"
} as const;

export const motionStagger = {
  compact: 0.04,
  section: 0.08
} as const;

export function configureMotionDefaults() {
  gsap.defaults({ duration: motionDurations.enter, ease: motionEase.standard, overwrite: "auto" });
  const media = gsap.matchMedia();
  media.add("(prefers-reduced-motion: reduce)", () => {
    gsap.defaults({ duration: 0.01, ease: "none", overwrite: "auto" });
    return () => gsap.defaults({ duration: motionDurations.enter, ease: motionEase.standard, overwrite: "auto" });
  });
  return media;
}

export function motionDuration(value: keyof typeof motionDurations, reducedMotion: boolean) {
  return reducedMotion ? 0.01 : motionDurations[value];
}
