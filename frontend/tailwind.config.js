/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#08090c",
          900: "#0b0e13",
          850: "#0f1319",
          800: "#131822",
          750: "#171d29",
          700: "#1e2635",
          600: "#2b3648",
        },
        lens: {
          DEFAULT: "#a3e635",
          dim: "#84cc16",
          soft: "#d9f99d",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        panel: "0 8px 30px rgba(0,0,0,0.35)",
      },
    },
  },
  plugins: [],
};
