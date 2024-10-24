import type { NavItemConfig } from '@/types/nav';
import { paths } from '@/paths';

export const navItems = [
  { key: 'overview', title: 'Overview', href: paths.dashboard.overview, icon: 'chart-pie' },
  { key: 'batch', title: 'Batch', href: paths.dashboard.batch, icon: 'chart-pie' },
  { key: 'settings', title: 'Settings', icon: 'gear-six' },
  { key: 'account', title: 'Account', icon: 'user' },
] satisfies NavItemConfig[];
