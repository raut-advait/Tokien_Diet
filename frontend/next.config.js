/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    let destination = process.env.BACKEND_API_URL || 'http://127.0.0.1:8000/api/:path*';
    if (process.env.BACKEND_API_URL && !destination.includes(':path*')) {
      if (destination.endsWith('/api') || destination.endsWith('/api/')) {
        destination = destination.replace(/\/$/, '') + '/:path*';
      } else {
        destination = destination.replace(/\/$/, '') + '/api/:path*';
      }
    }
    return [
      {
        source: '/api/:path*',
        destination: destination,
      },
    ];
  },
};

module.exports = nextConfig;
