import * as React from 'react';
import type { Metadata } from 'next';
import Grid from '@mui/material/Unstable_Grid2';
import dayjs from 'dayjs';

import { config } from '@/config';
import { Predict_traffic } from '@/components/dashboard/overview/tls-prediction';
import { Livepacket } from '@/components/dashboard/overview/live-packets';
import { Livepacketflow } from '@/components/dashboard/overview/live-packets-flow';
import { Traffic } from '@/components/dashboard/overview/traffic';
import { Application } from '@/components/dashboard/overview/Application';
import { height } from '@mui/system';

export const metadata = { title: `Overview | Dashboard | ${config.site.name}` } satisfies Metadata;

export default function Page(): React.JSX.Element {
  return (
    <Grid container spacing={2}>
      <Grid lg={6} xs={12} sx={{ height: '100%' }}>
        <Livepacket/>
      </Grid>
      <Grid lg={3} md={6} xs={12}>
        <Traffic chartSeries={[63, 15, 22]} labels={['Chat', 'Voip', 'Streaming']} sx={{ height: '100%'}}/>
      </Grid>
      <Grid lg={3} md={6} xs={12}>
        <Application chartSeries={[63, 15, 22]} labels={['Facebook', 'Discord', 'Skype','Line','Youtube']} sx={{ height: '100%' }} />
      </Grid>
      <Grid lg={12} md={6} xs={12} sx={{ height: '100%' }}>
        <Livepacketflow/>
      </Grid>
      <Grid lg={12} md={12} xs={12}>
        <Predict_traffic
          predicts={[
          ]}
          sx={{ height: '100%' }}
        />
      </Grid>
    </Grid>
  );
}
