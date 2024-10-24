'use client';
import * as React from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardHeader from '@mui/material/CardHeader';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import type { SxProps } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import { ArrowRight as ArrowRightIcon } from '@phosphor-icons/react/dist/ssr/ArrowRight';
import dayjs from 'dayjs';

export interface Predict_traffic {
  protocol: string;
  traffic: string;
  app: string;
  startAt: Date;
  endAt: Date;
}

export interface Predict_trafficProps {
  predicts?: Predict_traffic[];
  sx?: SxProps;
}

export function Predict_traffic({ predicts = [], sx }: Predict_trafficProps): React.JSX.Element {
  return (
    <Card sx={sx}>
      <CardHeader title="SSL/TLS traffic prediction" />
      <Divider />
      <Box sx={{ overflowX: 'auto' }}>
        <Table sx={{ minWidth: 800 }}>
          <TableHead>
            <TableRow>
              <TableCell>Protocol</TableCell>
              <TableCell sortDirection="desc">StartDate</TableCell>
              <TableCell sortDirection="desc">EndDate</TableCell>
              <TableCell>TrafficID</TableCell>
              <TableCell>ApplicationID</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {predicts && predicts.length > 0 ? (
              predicts.map((predict) => (
                <TableRow hover key={predict.protocol}>
                  <TableCell>{predict.protocol}</TableCell>
                  <TableCell>{dayjs(predict.startAt).format('MMM D, YYYY')}</TableCell>
                  <TableCell>{dayjs(predict.endAt).format('MMM D, YYYY')}</TableCell>
                  <TableCell>{predict.traffic}</TableCell>
                  <TableCell>{predict.app}</TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={5} align="center">No Predict available</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Box>
      <Divider />
      <CardActions sx={{ justifyContent: 'flex-end' }}>
        <Button
          color="inherit"
          endIcon={<ArrowRightIcon fontSize="var(--icon-fontSize-md)" />}
          size="small"
          variant="text"
        >
          View all
        </Button>
      </CardActions>
    </Card>
  );
}
