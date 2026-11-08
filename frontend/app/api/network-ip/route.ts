import { NextResponse } from "next/server";
import os from "os";

export async function GET() {
  const ifaces = os.networkInterfaces();
  const ips: { name: string; ip: string }[] = [];

  for (const name of Object.keys(ifaces)) {
    for (const net of ifaces[name] || []) {
      if (net.family === "IPv4" && !net.internal && !net.address.startsWith("169.254")) {
        ips.push({ name, ip: net.address });
      }
    }
  }

  // Prioritize Wi-Fi or physical Ethernet
  const primary =
    ips.find((i) => /wi-?fi|wlan/i.test(i.name)) ||
    ips.find((i) => /ethernet/i.test(i.name)) ||
    ips[0] ||
    { name: "localhost", ip: "localhost" };

  return NextResponse.json({
    primaryIp: primary.ip,
    interfaceName: primary.name,
    allInterfaces: ips,
  });
}

