import { ActiveFireEventCard } from "../components/fire-events/ActiveFireEventCard";
import { EmptyState } from "../components/feedback/EmptyState";
import { ErrorState } from "../components/feedback/ErrorState";
import { LoadingState } from "../components/feedback/LoadingState";
import { MetricCard } from "../components/data/MetricCard";
import { TimestampDisplay } from "../components/data/TimestampDisplay";
import { PageHeader } from "../components/layout/PageHeader";
import { Link } from "react-router-dom";
import { useActiveFireEvents } from "../hooks/useActiveFireEvents";
import "./ActiveWildfiresPage.css";

const PAGE_TITLE = "Active Wildfires";
const PAGE_DESCRIPTION = "Current suspected and confirmed wildfire events.";

/**
 * The real US 6.1 dashboard: fetches GET /api/v1/fire-events/active via
 * useActiveFireEvents() and renders summary metrics + a card per event.
 * Counting confirmed/suspected here is presentation aggregation over the
 * API response - it never calculates severity or FireEvent status.
 */
export function ActiveWildfiresPage() {
  const { data, isLoading, isRefreshing, loadError, refreshError, refresh } = useActiveFireEvents();

  if (isLoading) {
    return (
      <section>
        <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} />
        <LoadingState message="Loading active wildfire events…" />
      </section>
    );
  }

  if (loadError) {
    return (
      <section>
        <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} />
        <ErrorState title="Unable to load active wildfire events." message="Please try again." onRetry={refresh} />
      </section>
    );
  }

  const items = data?.items ?? [];
  const confirmedCount = items.filter((event) => event.status === "confirmed").length;
  const suspectedCount = items.filter((event) => event.status === "suspected").length;

  return (
    <section>
      <PageHeader
        title={PAGE_TITLE}
        description={PAGE_DESCRIPTION}
        actions={
          <>
            <Link to="/response-plan" className="active-wildfires-page__global-plan">
              Global Response Plan
            </Link>
            <button
              type="button"
              className="active-wildfires-page__refresh"
              onClick={refresh}
              disabled={isRefreshing}
            >
              {isRefreshing ? "Refreshing…" : "Refresh"}
            </button>
          </>
        }
      />

      {data ? (
        <p className="active-wildfires-page__as-of">
          Data as of: <TimestampDisplay value={data.as_of} />
        </p>
      ) : null}

      {refreshError ? (
        <p role="alert" className="active-wildfires-page__refresh-error">
          {refreshError}
        </p>
      ) : null}

      <div className="active-wildfires-page__metrics">
        <MetricCard label="Active Events" value={items.length} />
        <MetricCard label="Confirmed" value={confirmedCount} />
        <MetricCard label="Suspected" value={suspectedCount} />
      </div>

      {items.length === 0 ? (
        <EmptyState
          title="No active wildfire events"
          message="There are currently no suspected or confirmed wildfire events."
        />
      ) : (
        <section aria-labelledby="active-incidents-heading">
          <h2 id="active-incidents-heading" className="active-wildfires-page__section-title">
            Active incidents
          </h2>
          <div className="active-wildfires-page__grid">
            {items.map((event) => (
              <ActiveFireEventCard key={event.fire_event_id} event={event} />
            ))}
          </div>
        </section>
      )}
    </section>
  );
}
