import { useState } from "react";
import { ActiveFiresPanel } from "../components/dashboard/ActiveFiresPanel";
import { OperationsActivityDrawer } from "../components/dashboard/OperationsActivityDrawer";
import { OperationsActivityFeed } from "../components/dashboard/OperationsActivityFeed";
import { OperationsMap } from "../components/dashboard/OperationsMap";
import { OperationsStatusHeader } from "../components/dashboard/OperationsStatusHeader";
import { ErrorState } from "../components/feedback/ErrorState";
import { LoadingState } from "../components/feedback/LoadingState";
import { PageHeader } from "../components/layout/PageHeader";
import { Link } from "react-router-dom";
import { useDemoDataVisibility } from "../hooks/demoSession";
import { useOperationsOverview } from "../hooks/useOperationsOverview";
import type { OperationsActivityFeedItem } from "../types/operationsOverview";
import "./ActiveWildfiresPage.css";

const PAGE_TITLE = "Operations Overview";
export const NO_SIMULATION_TITLE = "No simulation started";
export const NO_SIMULATION_MESSAGE = "Start a simulation to begin monitoring.";

/**
 * The dashboard/Operations screen (US 6.1, wired to A6/A7 in Task A7,
 * redesigned map-first in Task A8): a single call to
 * `useOperationsOverview()` remains the sole data source for the whole
 * page - no direct polling of any sub-endpoint, and Activity Detail (A5)
 * is only fetched for the one item the operator selects
 * (`useOperationsActivityDetail`, inside `OperationsActivityDrawer`).
 *
 * Loading/error/empty states follow A7's contract: `isLoading` blanks the
 * page once (first load only); `loadError` is a page-level failure with
 * retry; a background `refreshError` keeps the last good snapshot visible
 * with a subtle inline warning, never blanking the map/panel/feed.
 *
 * Task A9: `OperationsStatusHeader` now also owns the one Start
 * Simulation/Run Again action. `refresh` (this same hook's own manual
 * refetch) is handed down as `onRequestOverviewRefresh` so a successful
 * start - or a 409/403 rejection - nudges one immediate overview refetch;
 * this page never starts a second polling loop or mirrors simulation state
 * itself.
 *
 * Layout polish: the top row keeps the map dominant with Active Fires to
 * its right (both equally tall, no forced inner scrolling on Active Fires
 * for the normal 1-2 fire demo case); the Activity Feed moves to its own
 * full-width row below, since a tall right-hand column made the page feel
 * asymmetric and cramped the feed's preview text.
 *
 * Initial demo state: runtime rows from a PREVIOUS demo run stay in Neon
 * until the next Start Simulation resets them, so `useDemoDataVisibility`
 * hides them (presentation only - nothing is deleted) until this browser
 * session starts or observes a run. See hooks/demoSession.ts.
 */
export function ActiveWildfiresPage() {
  const { data, isLoading, loadError, refreshError, refresh } = useOperationsOverview();
  const [selectedActivity, setSelectedActivity] = useState<OperationsActivityFeedItem | null>(null);
  const showDemoData = useDemoDataVisibility(data?.simulation ?? null);

  if (isLoading) {
    return (
      <section>
        <PageHeader title={PAGE_TITLE} />
        <LoadingState message="Loading operations overview…" />
      </section>
    );
  }

  if (loadError || !data) {
    return (
      <section>
        <PageHeader title={PAGE_TITLE} />
        <ErrorState title="Unable to load the operations overview." message="Please try again." onRetry={refresh} />
      </section>
    );
  }

  const activeFires = showDemoData ? data.active_fires : [];
  const fireDangerAreas = showDemoData ? data.fire_danger_areas : [];
  const activityItems = showDemoData ? data.activity_feed.items : [];

  return (
    <section>
      <PageHeader
        title={PAGE_TITLE}
        actions={
          <Link to="/response-plan" className="active-wildfires-page__global-plan">
            Global Response Plan
          </Link>
        }
      />

      <OperationsStatusHeader
        simulation={data.simulation}
        generatedAt={data.generated_at}
        refreshError={refreshError}
        onRequestOverviewRefresh={refresh}
      />

      <div className="active-wildfires-page__top-row">
        <div className="active-wildfires-page__map-column">
          <OperationsMap fireDangerAreas={fireDangerAreas} activeFires={activeFires} />
        </div>

        <div className="active-wildfires-page__fires-column">
          {showDemoData ? (
            <ActiveFiresPanel activeFires={activeFires} />
          ) : (
            <ActiveFiresPanel activeFires={[]} emptyTitle={NO_SIMULATION_TITLE} emptyMessage={NO_SIMULATION_MESSAGE} />
          )}
        </div>
      </div>

      <div className="active-wildfires-page__feed-row">
        <OperationsActivityFeed
          items={activityItems}
          selectedActivityId={selectedActivity?.activity_id ?? null}
          onSelectItem={(item) =>
            setSelectedActivity((current) => (current?.activity_id === item.activity_id ? null : item))
          }
        />
      </div>

      <OperationsActivityDrawer selectedItem={selectedActivity} onClose={() => setSelectedActivity(null)} />
    </section>
  );
}
