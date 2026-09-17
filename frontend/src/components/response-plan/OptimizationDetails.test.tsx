import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { OptimizationConfig, ResponsePlan } from "../../types/responsePlan";
import { OptimizationDetails } from "./OptimizationDetails";

function makeConfig(overrides: Partial<OptimizationConfig> = {}): OptimizationConfig {
  return {
    population_size: 50,
    generation_count: 100,
    mutation_rate: 0.1,
    crossover_rate: 0.8,
    eta_reference_seconds: 600,
    initial_assignment_probability: 0.5,
    tournament_size: 3,
    elitism_count: 2,
    ...overrides,
  };
}

function makePlanSlice(
  overrides: Partial<Pick<ResponsePlan, "methodology" | "methodology_version" | "random_seed" | "optimization_config">> = {},
): Pick<ResponsePlan, "methodology" | "methodology_version" | "random_seed" | "optimization_config"> {
  return {
    methodology: "genetic_algorithm",
    methodology_version: "1.0.0",
    random_seed: 42,
    optimization_config: makeConfig(),
    ...overrides,
  };
}

describe("OptimizationDetails", () => {
  it("displays the methodology", () => {
    render(<OptimizationDetails plan={makePlanSlice({ methodology: "genetic_algorithm" })} />);

    expect(screen.getByText("genetic_algorithm")).toBeInTheDocument();
  });

  it("displays the methodology version", () => {
    render(<OptimizationDetails plan={makePlanSlice({ methodology_version: "2.3.0" })} />);

    expect(screen.getByText("2.3.0")).toBeInTheDocument();
  });

  it("displays the random seed", () => {
    render(<OptimizationDetails plan={makePlanSlice({ random_seed: 12345 })} />);

    expect(screen.getByText("12345")).toBeInTheDocument();
  });

  it("displays the persisted population size", () => {
    render(<OptimizationDetails plan={makePlanSlice({ optimization_config: makeConfig({ population_size: 64 }) })} />);

    expect(screen.getByText("64")).toBeInTheDocument();
  });

  it("displays the persisted generation count", () => {
    render(
      <OptimizationDetails plan={makePlanSlice({ optimization_config: makeConfig({ generation_count: 200 }) })} />,
    );

    expect(screen.getByText("200")).toBeInTheDocument();
  });

  it("displays the persisted mutation rate", () => {
    render(<OptimizationDetails plan={makePlanSlice({ optimization_config: makeConfig({ mutation_rate: 0.15 }) })} />);

    expect(screen.getByText("0.15")).toBeInTheDocument();
  });

  it("displays the persisted crossover rate", () => {
    render(<OptimizationDetails plan={makePlanSlice({ optimization_config: makeConfig({ crossover_rate: 0.9 }) })} />);

    expect(screen.getByText("0.9")).toBeInTheDocument();
  });

  it("displays the persisted ETA reference seconds", () => {
    render(
      <OptimizationDetails
        plan={makePlanSlice({ optimization_config: makeConfig({ eta_reference_seconds: 500 }) })}
      />,
    );

    expect(screen.getByText("500")).toBeInTheDocument();
  });

  it("displays the persisted initial assignment probability", () => {
    render(
      <OptimizationDetails
        plan={makePlanSlice({ optimization_config: makeConfig({ initial_assignment_probability: 0.4 }) })}
      />,
    );

    expect(screen.getByText("0.4")).toBeInTheDocument();
  });

  it("displays the persisted tournament size", () => {
    render(<OptimizationDetails plan={makePlanSlice({ optimization_config: makeConfig({ tournament_size: 5 }) })} />);

    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("displays the persisted elitism count", () => {
    render(<OptimizationDetails plan={makePlanSlice({ optimization_config: makeConfig({ elitism_count: 4 }) })} />);

    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("shows a clear unavailable message when optimization_config is null, while still showing methodology/version/seed", () => {
    render(
      <OptimizationDetails
        plan={makePlanSlice({
          optimization_config: null,
          methodology: "genetic_algorithm",
          methodology_version: "1.0.0",
          random_seed: 42,
        })}
      />,
    );

    expect(
      screen.getByText("Detailed optimization configuration is not available for this response plan."),
    ).toBeInTheDocument();
    expect(screen.getByText("genetic_algorithm")).toBeInTheDocument();
    expect(screen.getByText("1.0.0")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("does not substitute default/hardcoded GA values when optimization_config is null", () => {
    render(<OptimizationDetails plan={makePlanSlice({ optimization_config: null })} />);

    expect(screen.queryByText("Population size")).not.toBeInTheDocument();
    expect(screen.queryByText("Generation count")).not.toBeInTheDocument();
    expect(screen.queryByText("Mutation rate")).not.toBeInTheDocument();
  });

  it("is collapsed by default and expands via an accessible native disclosure", () => {
    render(<OptimizationDetails plan={makePlanSlice()} />);

    const details = screen.getByText("Technical details").closest("details") as HTMLDetailsElement;
    expect(details).not.toBeNull();
    expect(details.open).toBe(false);
    expect(screen.getByText("Technical details").tagName.toLowerCase()).toBe("summary");
  });
});
