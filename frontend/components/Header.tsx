import Link from 'next/link';

export default function Header() {
  return (
    <header className="pt-6 px-[10%]">
      <div className="flex flex-col md:flex-row justify-between md:items-end pb-4">
        <div>
          <h1 className="text-6xl font-bold mb-1">
            White is for Witching: A Study into Narrative Voices
          </h1>
          <p className="text-2xl text-gray-600">Enbao Cao</p>
        </div>
        <nav className="flex space-x-8 mt-2 md:mt-0 text-xl">
          <Link href="#summary" className="text-gray-500 hover:text-gray-800 no-underline">Summary</Link>
          <Link href="#narrator" className="text-gray-500 hover:text-gray-800 no-underline">Narrators</Link>
          <Link href="#conclusion" className="px-3 py-1 hover:bg-gray-100 text-gray-500 hover:text-gray-800 no-underline">Conclusion</Link>
        </nav>
      </div>
    </header>
  );
}
