const fs = require('fs');
const path = require('path');
const os = require('os');
const selfsigned = require('selfsigned');

async function main() {
  const certDir = path.join(__dirname, 'certificates');
  if (!fs.existsSync(certDir)) {
    fs.mkdirSync(certDir, { recursive: true });
  }

  const altNames = [
    { type: 2, value: 'localhost' },
    { type: 7, ip: '127.0.0.1' },
    { type: 7, ip: '0.0.0.0' }
  ];

  const ifaces = os.networkInterfaces();
  for (const dev in ifaces) {
    if (ifaces[dev]) {
      for (const details of ifaces[dev]) {
        if (details.family === 'IPv4') {
          altNames.push({ type: 7, ip: details.address });
        }
      }
    }
  }

  const attrs = [{ name: 'commonName', value: 'synthesis-zero.local' }];
  const pems = await selfsigned.generate(attrs, {
    days: 365,
    keySize: 2048,
    algorithm: 'sha256',
    extensions: [{
      name: 'subjectAltName',
      altNames: altNames
    }]
  });

  const keyPath = path.join(certDir, 'server.key');
  const certPath = path.join(certDir, 'server.crt');

  fs.writeFileSync(keyPath, pems.private);
  fs.writeFileSync(certPath, pems.cert);

  console.log('SSL Certificates generated successfully:');
  console.log(' Key: ', keyPath);
  console.log(' Cert:', certPath);
}

main().catch(err => {
  console.error('Error generating certificate:', err);
  process.exit(1);
});
