import type { ReactNode } from "react";
import "./PageHeader.css";

export interface PageHeaderProps {
  title: string;
  description?: string;
  /** Optional slot for page-level actions such as a future Refresh button. */
  actions?: ReactNode;
}

/**
 * Reusable page-level heading, shared by every Epic 6 page (Active
 * Wildfires, Event Details, Response Plan, History, Monitoring). Renders a
 * single semantic <h1> - pages should not render their own duplicate <h1>.
 */
export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-header__text">
        <h1 className="page-header__title">{title}</h1>
        {description ? <p className="page-header__description">{description}</p> : null}
      </div>
      {actions ? <div className="page-header__actions">{actions}</div> : null}
    </header>
  );
}
