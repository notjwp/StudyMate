import Link from 'next/link';
import Image from 'next/image';
import {
  Bot,
  FileText,
  BookOpen,
  HelpCircle,
  Share2,
  ArrowRight,
  Clock,
  StickyNote,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { placeholderImages } from '@/lib/placeholder-images';
import LandingHeader from '@/components/layout/LandingHeader';

const features = [
  {
    href: '/qa',
    icon: Bot,
    label: 'PDF Q&A',
    description: 'Ask questions and get instant answers from your documents.',
  },
  {
    href: '/summarize',
    icon: FileText,
    label: 'Summarization',
    description: 'Generate quick or detailed summaries of your study materials.',
  },
  {
    href: '/flashcards',
    icon: BookOpen,
    label: 'Flashcards',
    description: 'Create interactive flashcards to test your knowledge.',
  },
  {
    href: '/quiz',
    icon: HelpCircle,
    label: 'Quiz',
    description: 'Generate quizzes from your documents to prepare for exams.',
  },
  {
    href: '/pomodoro',
    icon: Clock,
    label: 'Pomodoro Timer',
    description: 'Boost your productivity with focused work sessions.',
  },
  {
    href: '/notes',
    icon: StickyNote,
    label: 'Sticky Notes',
    description: 'Capture your ideas and thoughts on a digital note board.',
  },
  {
    href: '/concept-map',
    icon: Share2,
    label: 'Concept Map',
    description: 'Visualize connections and relationships between topics.',
  },
];

export default function RootPage() {
  const heroImage = placeholderImages.find(img => img.id === 'hero-background');

  return (
    <div className="flex flex-col min-h-screen">
       <LandingHeader />
       <main className="flex-1">
        <section className="relative w-full h-[60vh] flex items-center justify-center text-center">
          {heroImage && (
             <Image 
              src={heroImage.imageUrl}
              alt={heroImage.description}
              fill
              className="object-cover -z-10 brightness-50"
              data-ai-hint={heroImage.imageHint}
              priority
            />
          )}
          <div className="container px-4 md:px-6">
            <div className="max-w-3xl mx-auto">
              <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl md:text-6xl">
                Welcome to StudyMate
              </h1>
              <p className="mt-4 text-lg text-gray-200 md:text-xl">
                Your AI-powered study partner. Upload a document and let our AI help you learn smarter, not harder.
              </p>
              <div className="mt-8 flex justify-center">
                <Button asChild size="lg">
                  <Link href="/qa">Get Started <ArrowRight className="ml-2 h-5 w-5" /></Link>
                </Button>
              </div>
            </div>
          </div>
        </section>
        
        <section id="features" className="py-12 md:py-24">
          <div className="container px-4 md:px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">All-in-One Study Toolkit</h2>
              <p className="mt-3 max-w-2xl mx-auto text-lg text-muted-foreground">
                From asking questions to generating quizzes, we've got you covered.
              </p>
            </div>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {features.map((feature) => (
                <Link href={feature.href} key={feature.href}>
                  <Card className="group hover:shadow-lg transition-shadow h-full flex flex-col hover:border-primary">
                    <CardHeader className="flex flex-row items-center justify-between">
                       <div className="bg-primary/10 p-3 rounded-full">
                         <feature.icon className="h-6 w-6 text-primary" />
                       </div>
                    </CardHeader>
                    <CardContent className="flex-grow">
                      <CardTitle className="text-xl">{feature.label}</CardTitle>
                      <CardDescription className="mt-2">{feature.description}</CardDescription>
                    </CardContent>
                    <div className="p-6 pt-0">
                        <div className="flex items-center text-sm font-semibold text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                          Go to feature <ArrowRight className="ml-2 h-4 w-4" />
                        </div>
                    </div>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        </section>
      </main>
      <footer className="bg-muted py-6">
        <div className="container mx-auto text-center text-muted-foreground text-sm">
          © {new Date().getFullYear()} StudyMate. All rights reserved.
        </div>
      </footer>
    </div>
  );
}
