import * as React from 'react'
import { Outlet, useRouteError } from 'react-router'
import { requireUserWithRole } from '#app/utils/permissions.server.ts'
import { prisma } from '#app/utils/db.server.ts'
import { invariantResponse } from '@epic-web/invariant'
import { GeneralErrorBoundary } from '../../components/error-boundary.tsx'

export async function loader({ request }: { request: Request }) {
	const userId = await requireUserWithRole(request, 'admin')
	const user = await prisma.user.findUnique({ where: { id: userId } })
	invariantResponse(user, 'User not found', { status: 404 })
	return { user }
}

function ErrorBoundary() {
	const error = useRouteError() as any

	if (error?.status === 403) {
		return (
			<div className="container mx-auto flex h-full w-full flex-col justify-center pt-20 pb-32 text-center">
				<h3 className="text-h3">Unauthorized</h3>
				<p className="mt-2">You do not have access to this page.</p>
			</div>
		)
	}

	return <GeneralErrorBoundary />
}

export default function TrainingRoute() {
	return (
		<div className="m-auto mt-8 mb-24 max-w-3xl">
			<div className="container">
				<main className="bg-muted mx-auto px-6 py-8 md:container md:rounded-3xl">
					<h1 className="text-h1">Training</h1>
					<p className="text-body-md mt-4">
						Welcome to your training dashboard!
					</p>
					<Outlet />
				</main>
			</div>
		</div>
	)
}

export { ErrorBoundary }
