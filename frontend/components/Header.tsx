import Link from 'next/link';

export default function Header() {
  return (
    <header className="pt-6 px-[10%] border-b border-gray-300">
      <div className="flex flex-col md:flex-row justify-between md:items-end pb-4">
        <div>
          <h1 className="text-6xl font-bold mb-1">
            White is for Witching: Narrator Analysis
          </h1>
          <p className="text-2xl text-gray-600">By XX</p>
        </div>
        <nav className="space-x-4 mt-2 md:mt-0 text-xl">
          <Link href="#summary" className="text-gray-700 hover:text-black">Summary</Link>
          <Link href="#narrators" className="text-gray-700 hover:text-black">Narrators</Link>
          <Link href="#about" className="px-3 py-1 border border-gray-400 rounded hover:bg-gray-100">About</Link>
        </nav>
      </div>
    </header>
  );
}
