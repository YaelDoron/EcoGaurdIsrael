import "./DispatchNavigation.css";

type LatLon = { latitude: number; longitude: number };

export interface DispatchNavigationProps {
  /** The target's latitude/longitude - the navigation destination. */
  destination: LatLon | null;
  /** The resource's starting point (its station). Used as the Google Maps
   * `origin` so the route shown is station -> target, not from the
   * dispatcher's own location. Omitted from the link when unknown. */
  origin?: LatLon | null;
}

/** External deep links; the destination is the response target. */
/**
 * Waze Live Map directions (works from a desktop browser): explicit start and
 * end, so the route is station -> target rather than from the viewer's GPS.
 * Waze's `ll.` values are latitude,longitude (the same order as its other
 * deep links) - not longitude,latitude. `from` is omitted when the station's
 * coordinates are unknown, leaving Live Map to ask for a start.
 */
export function wazeUrl(destination: LatLon, origin: LatLon | null = null): string {
  const from = origin ? `from=ll.${origin.latitude},${origin.longitude}&` : "";
  return `https://www.waze.com/live-map/directions?${from}to=ll.${destination.latitude},${destination.longitude}`;
}

export function googleMapsUrl(destination: LatLon, origin: LatLon | null = null): string {
  const originParam = origin ? `&origin=${origin.latitude},${origin.longitude}` : "";
  return `https://www.google.com/maps/dir/?api=1${originParam}&destination=${destination.latitude},${destination.longitude}`;
}

function NavigationArrowIcon() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
      <path d="M8 0 1.5 15 8 11.8 14.5 15z" />
    </svg>
  );
}

function GeoAltIcon() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
      <path d="M8 16s6-5.686 6-10A6 6 0 0 0 2 6c0 4.314 6 10 6 10m0-7a3 3 0 1 1 0-6 3 3 0 0 1 0 6" />
    </svg>
  );
}

/**
 * "Dispatch & Navigation" panel: hands the destination to a dedicated
 * real-time navigation app instead of rendering route text. Both actions are
 * plain external links (new tab), so no routing data is fetched or computed
 * here. Both links carry the station as the explicit origin. Bootstrap is not part of this project, so the buttons use the app's
 * own light-theme styling.
 */
export function DispatchNavigation({ destination, origin = null }: DispatchNavigationProps) {
  return (
    <section className="dispatch-navigation" aria-label="Dispatch and navigation">
      <h4 className="dispatch-navigation__title">Dispatch &amp; Navigation</h4>
      {destination === null ? (
        <p className="dispatch-navigation__note">Navigation unavailable: this target has no coordinates.</p>
      ) : (
        <>
          <p className="dispatch-navigation__prompt">Launch real-time navigation:</p>
          <div className="dispatch-navigation__actions">
            <a
              className="dispatch-navigation__button dispatch-navigation__button--waze"
              href={wazeUrl(destination, origin)}
              target="_blank"
              rel="noopener noreferrer"
            >
              <NavigationArrowIcon />
              Navigate with Waze
            </a>
            <a
              className="dispatch-navigation__button dispatch-navigation__button--google"
              href={googleMapsUrl(destination, origin)}
              target="_blank"
              rel="noopener noreferrer"
            >
              <GeoAltIcon />
              Open Google Maps
            </a>
          </div>
        </>
      )}
    </section>
  );
}
