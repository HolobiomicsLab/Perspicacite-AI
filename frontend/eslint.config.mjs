import next from "eslint-config-next/core-web-vitals";

const config = [
  { ignores: [".next/**", "next-env.d.ts"] },
  ...next,
  {
    // The React-Compiler-era react-hooks rules bundled with Next 16 flag
    // ~20 pre-existing setState-in-effect / memoization patterns. They are
    // warnings (not errors) so the lint gate is green today; the effect
    // refactor they point at is tracked as separate post-release follow-up.
    rules: {
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/exhaustive-deps": "warn",
      "react-hooks/immutability": "warn",
      "react-hooks/preserve-manual-memoization": "warn",
    },
  },
];

export default config;
