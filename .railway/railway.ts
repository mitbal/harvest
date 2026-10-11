// Railway Infrastructure as Code for the data pipeline services. Replaces the
// railway.*.toml Config as Code files, which Railway stops reading on 2026-12-01.
//
// This is a named partial: it owns only the services declared here. The website
// (`harvest`) and Redis are managed in the Railway dashboard and left alone.
//
//   railway config plan     # read-only preview
//   railway config apply    # plan, confirm, apply
import { defineRailway, github, preserve, project, service } from "railway/iac";

export const partial = "pipeline";

const source = github("mitbal/harvest", { branch: "master" });

const pipelineBuild = (extraWatch: string[] = []) => ({
  builder: "DOCKERFILE" as const,
  dockerfilePath: "Dockerfile.pipeline",
  watchPatterns: [
    "harvest/**",
    "pipeline/**",
    ...extraWatch,
    "requirements-pipeline.txt",
    "Dockerfile.pipeline",
  ],
});

// Container limits as set in Railway; leaving them out would reset them.
const limits = (cpu: number, memoryGB: number) => ({
  containers: { cpu, memoryBytes: memoryGB * 1_000_000_000 },
});

// Values stay in Railway; set or change them in the dashboard.
const pipelineEnv = {
  FMP_API_KEY: preserve(),
  R2_ACCESS_KEY_ID: preserve(),
  R2_ACCOUNT_ID: preserve(),
  R2_BUCKET: preserve(),
  R2_SECRET_ACCESS_KEY: preserve(),
  REDIS_URL: preserve(),
  SUPABASE_KEY: preserve(),
  SUPABASE_URL: preserve(),
};

export default defineRailway(() => {
  // Nightly, 23:00 WIB on weekdays: prices, then the trading scans and track record.
  const jkse = service("daily-panen-jkse", {
    source,
    build: pipelineBuild(["data/jkse/syariah.csv"]),
    deploy: {
      startCommand: "sh pipeline/trading_nightly.sh",
      cronSchedule: "0 16 * * 1-5",
      restartPolicyType: "NEVER",
      limitOverride: limits(1, 2),
    },
    env: pipelineEnv,
  });

  // S&P 500 prices, 11:00 WIB Tuesday to Saturday (after the US close).
  const sp500 = service("daily-panen-sp500", {
    source,
    build: pipelineBuild(),
    deploy: {
      startCommand:
        "timeout --signal=TERM --kill-after=60s 6h python -m pipeline.daily_data --exchange sp500",
      cronSchedule: "0 4 * * 2-6",
      restartPolicyType: "NEVER",
      limitOverride: limits(1, 2),
    },
    env: pipelineEnv,
  });

  // Weekly setup search for every trading tool, Sunday 10:00 WIB. Railway stops a run after 6 hours.
  const optimize = service("trading-optimize", {
    source,
    build: pipelineBuild(),
    deploy: {
      startCommand: "sh pipeline/trading_weekly.sh",
      cronSchedule: "0 3 * * 0",
      restartPolicyType: "NEVER",
      limitOverride: limits(2, 3),
    },
    env: pipelineEnv,
  });

  return project("Harvest Dividend Website", { resources: [jkse, sp500, optimize] });
});
