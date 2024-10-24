const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      pathRewrite: {
        '/api': '/',  // '/api/batch'를 '/batch'로 변경하여 매핑
      },
      changeOrigin: true,
    })
  );

  // Sub 서버와 연결
  app.use(
    '/api/batch',
    createProxyMiddleware({
      target: 'http://localhost:5000',  // Sub 서버가 5001 포트에서 실행된다고 가정
      pathRewrite: {
        '/api/batch': '/batch',  // '/api/batch'를 '/batch'로 변경하여 매핑
      },
      changeOrigin: true,
    })
  );

  app.use(
    '/api/history',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      pathRewrite: {
        '/api/history': '/history',  // '/api/batch'를 '/batch'로 변경하여 매핑
      },
      changeOrigin: true,
    })
  );
};
