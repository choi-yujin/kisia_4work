export const paths = {
  home: '/',
  auth: { signIn: '/auth/sign-in', signUp: '/auth/sign-up', resetPassword: '/auth/reset-password' },
  dashboard: {
    overview: '/dashboard',
    batch: '/dashboard/batch'
    // account: '/dashboard/account',
    // settings: '/dashboard/settings',
  },
  errors: { notFound: '/errors/not-found' },
} as const;
