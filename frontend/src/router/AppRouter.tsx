import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import { ActiveWildfiresPage } from "../pages/ActiveWildfiresPage";
import { EventDetailsPage } from "../pages/EventDetailsPage";
import { NotFoundPage } from "../pages/NotFoundPage";
import { ResponsePlanPlaceholderPage } from "../pages/ResponsePlanPlaceholderPage";

/**
 * The application's route table. Generic on purpose (Task 15): it only
 * wires paths to pages inside the shared `AppShell` layout. Feature
 * behavior belongs in the page components themselves, not here.
 *
 * Exported separately from `main.tsx`'s `<BrowserRouter>` so tests can
 * render `<AppRouter />` inside a `<MemoryRouter>` with a chosen initial
 * route instead.
 */
export function AppRouter() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/events" replace />} />
        <Route path="/events" element={<ActiveWildfiresPage />} />
        <Route path="/events/:fireEventId" element={<EventDetailsPage />} />
        <Route path="/events/:fireEventId/plan" element={<ResponsePlanPlaceholderPage />} />
        <Route path="/plans/:planId" element={<ResponsePlanPlaceholderPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
