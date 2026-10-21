export function LogoMark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <linearGradient id="lg" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#22D3EE" />
          <stop offset="0.55" stopColor="#A78BFA" />
          <stop offset="1" stopColor="#F472B6" />
        </linearGradient>
      </defs>
      <g fill="none" stroke="url(#lg)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M6 20V10a4 4 0 0 1 4-4h10" />
        <path d="M44 6h10a4 4 0 0 1 4 4v10" />
        <path d="M58 44v10a4 4 0 0 1-4 4H44" />
        <path d="M20 58H10a4 4 0 0 1-4-4V44" />
      </g>
      <path d="M16 44C22 44 24 30 32 30S42 22 48 20" fill="none" stroke="url(#lg)" strokeWidth="3.2" strokeLinecap="round" />
      <circle cx="16" cy="44" r="2.6" fill="#22D3EE" opacity="0.55" />
      <circle cx="32" cy="30" r="3" fill="#A78BFA" opacity="0.85" />
      <circle cx="48" cy="20" r="4.6" fill="#F472B6" />
    </svg>
  );
}
