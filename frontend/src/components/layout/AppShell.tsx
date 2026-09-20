import { NavLink, Outlet } from "react-router-dom";
import "./AppShell.css";

/**
 * Reusable application shell (header + navigation + content outlet) shared
 * by every Epic 6 page. Feature pages render inside `<Outlet />` via the
 * router (see `src/router/AppRouter.tsx`) - this component stays generic
 * and must not gain US 6.1-specific content (Task 15).
 */
export function AppShell() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__title">EcoGuard Israel</span>
          <span className="app-header__subtitle">Wildfire Operations</span>
        </div>
        <nav className="app-nav" aria-label="Main">
          <ul className="app-nav__list">
            <li>
              <NavLink
                to="/events"
                className={({ isActive }) => `nav-link${isActive ? " nav-link--active" : ""}`}
              >
                Operations Overview
              </NavLink>
            </li>
          </ul>
        </nav>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
