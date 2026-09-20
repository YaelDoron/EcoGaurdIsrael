import { useEffect, useRef, useState } from "react";
import { EmptyState } from "../feedback/EmptyState";
import { NEW_ACTIVITY_HIGHLIGHT_MS } from "./activityFeedPresentation";
import { OperationsActivityItem } from "./OperationsActivityItem";
import type { OperationsActivityFeedItem } from "../../types/operationsOverview";
import "./OperationsActivityFeed.css";

export interface OperationsActivityFeedProps {
  items: OperationsActivityFeedItem[];
  selectedActivityId: string | null;
  onSelectItem: (item: OperationsActivityFeedItem) => void;
}

const INITIAL_VISIBLE_COUNT = 5;
const LOAD_MORE_STEP = 5;

/**
 * Final visible-feed semantics: the dashboard Activity Feed shows ONLY
 * `news_report` / `satellite_hotspot` / `fire_danger`. `fire_event` and
 * `fire_severity` are excluded because FireEvent status and Severity are
 * already clearly shown on the map, Active Fires cards, and Event Details -
 * repeating every lifecycle/reassessment as its own feed row was
 * noisy/repetitive. `global_planning_run` is excluded because it is now
 * reached via the dedicated "View Response Plan" action next to Active
 * Fires (see ViewResponsePlanAction). All three are filtered out of the
 * "eligible" candidate list BEFORE the visible-count window is applied, so
 * none of them ever occupies one of the 5 initially-visible slots and none
 * counts toward "Showing X of Y" - this is a frontend presentation filter
 * only; FireEvent/FireSeverityAssessment/GlobalPlanningRun persistence, A5
 * detail support, and A6 itself are all untouched.
 */
function isEligibleForFeed(item: OperationsActivityFeedItem): boolean {
  return (
    item.activity_type !== "global_planning_run" &&
    item.activity_type !== "fire_severity" &&
    item.activity_type !== "fire_event"
  );
}

/**
 * The Operations Overview's recent-activity timeline: a true row list (one
 * activity per horizontal row - see OperationsActivityItem), rendering A6's
 * `activity_feed.items` in EXACT server order (already `available_at DESC`)
 * - this component never sorts, filters by anything other than
 * `isEligibleForFeed`, or groups items. Selecting an item is
 * presentation-only state lifted to the parent, which drives
 * `useOperationsActivityDetail` for exactly the one selected item (never
 * one fetch per rendered item).
 *
 * Visible-count windowing (client-side only - no second request): only the
 * first `visibleCount` eligible, server-ordered items are shown, starting
 * at 5. `visibleCount` is plain component state that only ever grows via
 * "Load more" - a poll returning a new snapshot never resets it back to 5,
 * and never auto-grows it either. Because the server always places newer
 * activity_ids first, a newly-arrived eligible item lands at index 0 and
 * the slice naturally pushes whatever was previously last inside the
 * window out of view - no special "insert at top" logic is needed here.
 *
 * New-arrival highlighting: as newer overview snapshots arrive via `items`
 * prop changes (A7's own polling - this component starts no fetch/timer of
 * its own), any eligible `activity_id` not present in the previous
 * snapshot is briefly marked "new" - a presentation-only diff of stable
 * backend ids, never a locally fabricated activity. The very first
 * snapshot this component ever sees marks nothing as new (there is no
 * "previous" to compare against).
 */
export function OperationsActivityFeed({ items, selectedActivityId, onSelectItem }: OperationsActivityFeedProps) {
  const eligibleItems = items.filter(isEligibleForFeed);

  const previousIdsRef = useRef<Set<string> | null>(null);
  const [newlyArrivedIds, setNewlyArrivedIds] = useState<Set<string>>(new Set());
  const pendingTimeoutsRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_COUNT);

  useEffect(() => {
    const currentIds = new Set(eligibleItems.map((item) => item.activity_id));
    const previousIds = previousIdsRef.current;
    previousIdsRef.current = currentIds;

    if (previousIds === null) {
      // First snapshot this component has ever seen - nothing is "new" yet.
      return;
    }

    const arrivedIds = eligibleItems.map((item) => item.activity_id).filter((id) => !previousIds.has(id));
    if (arrivedIds.length === 0) {
      return;
    }

    setNewlyArrivedIds((current) => {
      const next = new Set(current);
      for (const id of arrivedIds) {
        next.add(id);
      }
      return next;
    });

    const timeoutId = setTimeout(() => {
      pendingTimeoutsRef.current.delete(timeoutId);
      setNewlyArrivedIds((current) => {
        const next = new Set(current);
        for (const id of arrivedIds) {
          next.delete(id);
        }
        return next;
      });
    }, NEW_ACTIVITY_HIGHLIGHT_MS);
    pendingTimeoutsRef.current.add(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items]);

  useEffect(
    () => () => {
      for (const timeoutId of pendingTimeoutsRef.current) {
        clearTimeout(timeoutId);
      }
    },
    [],
  );

  const visibleItems = eligibleItems.slice(0, visibleCount);
  const hasMore = eligibleItems.length > visibleItems.length;

  return (
    <section aria-labelledby="operations-activity-feed-heading" className="operations-activity-feed">
      {eligibleItems.length === 0 ? (
        <>
          <h2 id="operations-activity-feed-heading" className="operations-activity-feed__title">
            Activity Feed
          </h2>
          <EmptyState title="No recent operational activity" message="Activity will appear here as it happens." />
        </>
      ) : (
        <div className="operations-activity-feed__panel">
          <div className="operations-activity-feed__panel-header">
            <h2 id="operations-activity-feed-heading" className="operations-activity-feed__title">
              Activity Feed
            </h2>
            <span className="operations-activity-feed__count">
              Showing {visibleItems.length} of {eligibleItems.length}
            </span>
          </div>
          <ul className="operations-activity-feed__list">
            {visibleItems.map((item) => (
              <OperationsActivityItem
                key={item.activity_id}
                item={item}
                isSelected={item.activity_id === selectedActivityId}
                isNew={newlyArrivedIds.has(item.activity_id)}
                onSelect={onSelectItem}
              />
            ))}
          </ul>
          {hasMore ? (
            <div className="operations-activity-feed__footer">
              <button
                type="button"
                className="operations-activity-feed__load-more"
                onClick={() => setVisibleCount((count) => count + LOAD_MORE_STEP)}
              >
                Load more
              </button>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
