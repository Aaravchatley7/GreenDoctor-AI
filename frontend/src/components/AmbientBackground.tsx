import React, { memo } from 'react';

/**
 * AmbientBackground — Dynamic living atmosphere layer for Glassmorphism UI.
 *
 * Renders multiple GPU-accelerated blurred light orbs that slowly and organically
 * drift behind frosted glass panels, producing optical depth and subtle refraction
 * in both dark and light modes.
 */
export const AmbientBackground: React.FC = memo(() => {
  return (
    <div className="ambient-background-mesh" aria-hidden="true">
      {/* Orb 1 — Botanical Emerald / Mint Core */}
      <div className="ambient-orb orb-1" />
      {/* Orb 2 — Azure / Cyan Core */}
      <div className="ambient-orb orb-2" />
      {/* Orb 3 — Violet / Indigo Depth Core */}
      <div className="ambient-orb orb-3" />
      {/* Orb 4 — Teal / Sapphire Accent Node */}
      <div className="ambient-orb orb-4" />
      {/* Ambient Diffusion Veil */}
      <div className="ambient-diffusion-veil" />
    </div>
  );
});

AmbientBackground.displayName = 'AmbientBackground';
export default AmbientBackground;
