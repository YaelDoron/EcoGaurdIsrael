/**
 * Small, domain-agnostic type helpers shared across the frontend foundation.
 *
 * Feature-specific contracts (e.g. FireEvent shapes) belong to the feature
 * that introduces them (US 6.1's dashboard work, Task 6), not here.
 */
export type Nullable<T> = T | null;
